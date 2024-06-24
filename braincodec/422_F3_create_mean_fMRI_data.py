# evaluate 用のデータセットを読み込んで，正しく予測できたデータの内，一番自信もって予測できた bold を画像として出力する．
# layer ごとにも出力する.
import argparse
import typing as tp
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import torch
from nilearn import plotting
from nilearn.datasets import fetch_atlas_difumo
from nilearn.image import iter_img
from nilearn.maskers import NiftiMapsMasker
from tqdm import tqdm

from audiocraft.data.fMRI_dataset import SegmentInfo
from audiocraft.models.DownstreamRawCSM import DownstreamRawCSM
from audiocraft.solvers import builders
from audiocraft.solvers.compression_fMRI import CompressionfMRISolver

TASK2CORDS = {
    "EMOTION": (-20, -6, 16),
    "GAMBLING": (14, 6, 0),
    "LANGUAGE": (-52, -2, -8),
    "MOTOR": (8, -16, -20),
    "RELATIONAL": (40, 40, 16),
    "SOCIAL": (50, -56, 18),
    "WM": (-38, -16, 38),
    "REST": (0, -22, 16),
}


def tr_preprocess(
    max_tr: float,
    tr_precision: float,
    raw_tr: tp.List[float],
    max_durtion: int,
    bold_durations: tp.List[int],
) -> torch.Tensor:
    assert len(raw_tr) == len(bold_durations)

    tr_emb_pad_id = int(max_tr / tr_precision) + 1

    tr = torch.zeros(len(raw_tr), max_durtion).long()
    for idx, (_tr, bold_d) in enumerate(zip(raw_tr, bold_durations)):
        _tr_tensor = torch.arange(0, bold_d) * _tr
        _tr_tensor = (_tr_tensor / tr_precision).floor()
        if bold_d > max_durtion:
            _tr_tensor = _tr_tensor[:max_durtion]
        else:
            pad_length = max_durtion - bold_d
            _tr_tensor = torch.cat((_tr_tensor, tr_emb_pad_id * torch.ones(pad_length)))
        tr[idx] = _tr_tensor

    assert (tr <= tr_emb_pad_id).all()
    return tr


def fMRI_write(rec_bold, output_path, maps_masker, tr, task, block_num=2):
    rec_bold = rec_bold.cpu().numpy().T

    time_step = rec_bold.shape[0]
    block_num = time_step

    reconstructed_img = maps_masker.inverse_transform(rec_bold)

    fig, axes = plt.subplots(figsize=(20, block_num * 2), ncols=2, nrows=block_num // 2 + block_num % 2)
    axes = axes.reshape(-1)
    for i, cmap in enumerate(iter_img(reconstructed_img)):
        plotting.plot_stat_map(
            cmap,
            figure=fig,
            axes=axes[i],
            colorbar=False,
            title=f"t={i*tr:.2f}",
            cut_coords=TASK2CORDS[task],
        )

    plt.savefig(output_path)
    plt.close()


@torch.no_grad()
def _preprocess_tr(
    batch: tp.Tuple[torch.Tensor, SegmentInfo], device: str, return_raw_tr=False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess training batch."""
    x, infos = batch
    x = x.to(device)
    tr = torch.zeros(x.size(0), x.size(-1), device=device).long()
    raw_trs = []
    bold_durations = []
    labels = []
    names = []
    for i, info in enumerate(infos):
        _tr = info.tr
        _name = Path(info.meta.path).stem
        labels.append(info.meta.label)
        raw_trs.append(_tr)
        fMRI_duration = info.duration
        bold_durations.append(fMRI_duration)
        _tr_tensor = torch.arange(0, fMRI_duration, device=device) * _tr

        if fMRI_duration > x.size(-1):
            _tr_tensor = _tr_tensor[: x.size(-1)]
        else:
            pad_length = x.size(-1) - fMRI_duration
            pad_id = info.max_tr + info.tr_precision
            _tr_tensor = torch.cat((_tr_tensor, pad_id * torch.ones(pad_length, device=device)))

        _tr_tensor = (_tr_tensor / info.tr_precision).floor()

        names.append(_name)
        tr[i] = _tr_tensor

    labels = torch.tensor(labels, device=device).long()

    if return_raw_tr is True:
        return x, tr, raw_trs, bold_durations, labels, names
    return x, tr, labels


@torch.no_grad()
def get_reconstructed_fMRI(codec_model, use_layer, bold: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    output = codec_model.encode(bold, tr)
    if isinstance(output, torch.Tensor):
        rec = codec_model.decode(output, tr, noise_disable=True, use_layer=use_layer)
    elif len(output) == 2:
        codes, logvar = output
        rec = codec_model.decode(codes, tr, logvar, noise_disable=True, use_layer=use_layer)
    elif len(output) == 3:
        codes, mu, logvar = output
        rec = codec_model.decode(codes, tr, mu, logvar, noise_disable=True, use_layer=use_layer)
    else:
        raise RuntimeError("Only EncodecfMRIModel model is supported.")
    return rec, codes


def calculate_l1_dist(bold1, bold2):
    l1_dist = torch.mean(torch.abs(bold1 - bold2), dim=0)
    return l1_dist


def evaluate(
    dataloaders,
    codec_model,
    vae_model,
    target_run_name,
    label2task,
    maps_masker,
    max_seq_len=10,
    target_label=9,
    device="cpu",
):
    loader = dataloaders["evaluate"]
    target_data = torch.zeros(1024, max_seq_len)
    used_names = []
    target_bold_durations = []

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        bold, tr, raw_trs, bold_durations, labels, names = _preprocess_tr(batch, "cpu", return_raw_tr=True)
        for idx, label in enumerate(labels):
            if label.item() == target_label and names[idx] not in used_names:
                _bold = bold[idx]
                _tr = raw_trs[idx]
                if _tr > 0.719999 and _tr < 0.720001:
                    target_data += _bold[:, :max_seq_len]
                    used_names.append(names[idx])
                    target_bold_durations.append(bold_durations[idx])
                if names[idx] == target_run_name:
                    gt_data = _bold[:, :max_seq_len]
                    rec_full_codec, _ = get_reconstructed_fMRI(
                        codec_model,
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        _bold.unsqueeze(0).to(device),
                        tr[idx].unsqueeze(0).to(device),
                    )
                    rec_half_codec, _ = get_reconstructed_fMRI(
                        codec_model,
                        [
                            0,
                            1,
                            2,
                            3,
                        ],
                        _bold.unsqueeze(0).to(device),
                        tr[idx].unsqueeze(0).to(device),
                    )
                    rec_first_codec, _ = get_reconstructed_fMRI(
                        codec_model,
                        [
                            0,
                        ],
                        _bold.unsqueeze(0).to(device),
                        tr[idx].unsqueeze(0).to(device),
                    )
                    rec_vae, _ = get_reconstructed_fMRI(
                        vae_model,
                        None,
                        _bold.unsqueeze(0).to(device),
                        tr[idx].unsqueeze(0).to(device),
                    )
    task = label2task[target_label]
    target_data /= len(used_names)
    output_path = output_dir / f"{task}_mean_first_{max_seq_len}steps_label={target_label}_cnt={len(used_names)}.png"
    print("durations: ", set(target_bold_durations))
    gt_corr = calculate_l1_dist(target_data, gt_data)
    print(gt_corr)
    print(
        "Time-averaged l1 mean vs gt:",
        torch.mean(gt_corr).item(),
        "+-",
        torch.std(gt_corr).item(),
    )
    codec_full_corr = calculate_l1_dist(target_data, rec_full_codec[0, :, :max_seq_len].to("cpu"))
    print(
        "Time-averaged l1 mean vs full codec:",
        torch.mean(codec_full_corr).item(),
        "+-",
        torch.std(codec_full_corr).item(),
    )
    codec_half_corr = calculate_l1_dist(target_data, rec_half_codec[0, :, :max_seq_len].to("cpu"))
    print(
        "Time-averaged l1 mean vs half codec:",
        torch.mean(codec_half_corr).item(),
        "+-",
        torch.std(codec_half_corr).item(),
    )
    codec_first_corr = calculate_l1_dist(target_data, rec_first_codec[0, :, :max_seq_len].to("cpu"))
    print(
        "Time-averaged l1 mean vs first codec:",
        torch.mean(codec_first_corr).item(),
        "+-",
        torch.std(codec_first_corr).item(),
    )
    vae_corr = calculate_l1_dist(target_data, rec_vae[0, :, :max_seq_len].to("cpu"))
    print(
        "Time-averaged l1 mean vs vae:",
        torch.mean(vae_corr).item(),
        "+-",
        torch.std(vae_corr).item(),
    )
    fMRI_write(
        target_data,
        output_path,
        maps_masker,
        _tr,
        task,
    )

    l1_dists_gt = []
    l1_dists_full_codec = []
    l1_dists_half_codec = []
    l1_dists_first_codec = []
    l1_dists_vae = []
    used_names = []
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        bold, tr, raw_trs, bold_durations, labels, names = _preprocess_tr(batch, "cpu", return_raw_tr=True)
        for idx, label in enumerate(labels):
            if label.item() == target_label and names[idx] not in used_names:
                _bold = bold[idx]
                _tr = raw_trs[idx]
                gt_data = _bold[:, :max_seq_len]
                rec_full_codec, _ = get_reconstructed_fMRI(
                    codec_model,
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    _bold.unsqueeze(0).to(device),
                    tr[idx].unsqueeze(0).to(device),
                )
                rec_half_codec, _ = get_reconstructed_fMRI(
                    codec_model,
                    [
                        0,
                        1,
                        2,
                        3,
                    ],
                    _bold.unsqueeze(0).to(device),
                    tr[idx].unsqueeze(0).to(device),
                )
                rec_first_codec, _ = get_reconstructed_fMRI(
                    codec_model,
                    [
                        0,
                    ],
                    _bold.unsqueeze(0).to(device),
                    tr[idx].unsqueeze(0).to(device),
                )
                rec_vae, _ = get_reconstructed_fMRI(
                    vae_model,
                    None,
                    _bold.unsqueeze(0).to(device),
                    tr[idx].unsqueeze(0).to(device),
                )
                l1_dists_gt.append(torch.mean(calculate_l1_dist(target_data, gt_data)).item())
                l1_dists_full_codec.append(
                    torch.mean(calculate_l1_dist(target_data, rec_full_codec[0, :, :max_seq_len].to("cpu"))).item()
                )
                l1_dists_half_codec.append(
                    torch.mean(calculate_l1_dist(target_data, rec_half_codec[0, :, :max_seq_len].to("cpu"))).item()
                )
                l1_dists_first_codec.append(
                    torch.mean(calculate_l1_dist(target_data, rec_first_codec[0, :, :max_seq_len].to("cpu"))).item()
                )
                l1_dists_vae.append(
                    torch.mean(calculate_l1_dist(target_data, rec_vae[0, :, :max_seq_len].to("cpu"))).item()
                )
    print("l1_dists_gt:", np.mean(l1_dists_gt), np.std(l1_dists_gt))
    print(
        "l1_dists_full_codec:",
        np.mean(l1_dists_full_codec),
        np.std(l1_dists_full_codec),
    )
    print(
        "l1_dists_half_codec:",
        np.mean(l1_dists_half_codec),
        np.std(l1_dists_half_codec),
    )
    print(
        "l1_dists_first_codec:",
        np.mean(l1_dists_first_codec),
        np.std(l1_dists_first_codec),
    )
    print("l1_dists_vae:", np.mean(l1_dists_vae), np.std(l1_dists_vae))
    return


if __name__ == "__main__":
    # This code is only for CSM model
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_model_path", type=str, help="Path to classifier model checkpoint", required=True)
    parser.add_argument("--vae_codec_model_path", type=str, help="Path to VAE model checkpoint", required=True)
    parser.add_argument("--output_dir", type=str, help="Output directory", required=True)
    parser.add_argument(
        "--target_run_name", type=str, help="Target run name (the exact filename of HCP data)", required=True
    )
    parser.add_argument("--max_seq_len", type=int, help="Max sequence length", default=14)
    parser.add_argument("--target_label", type=int, help="Target label", default=9)
    args = parser.parse_args()

    label2task = {
        0: "EMOTION",
        1: "EMOTION",
        2: "GAMBLING",
        3: "GAMBLING",
        4: "LANGUAGE",
        5: "LANGUAGE",
        6: "MOTOR",
        7: "MOTOR",
        8: "MOTOR",
        9: "MOTOR",
        10: "MOTOR",
        11: "RELATIONAL",
        12: "RELATIONAL",
        13: "SOCIAL",
        14: "SOCIAL",
        15: "WM",
        16: "WM",
        17: "WM",
        18: "WM",
        19: "REST",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    classifier_model = DownstreamRawCSM.get_pretrained(args.classifier_model_path)
    encodec_model = classifier_model.compression_model
    vae_model = CompressionfMRISolver.model_from_checkpoint(args.vae_codec_model_path).to(device)
    cfg = classifier_model.lm.cfg
    cfg["execute_only"] = "evaluate"
    print("Loading datasets...")
    dataloaders = builders.get_fMRI_datasets(cfg)
    del classifier_model  # 用済み

    print("Preparing maps masker...")
    difumo_path = fetch_atlas_difumo(dimension=cfg.space_dim, resolution_mm=2, legacy_format=False).maps
    maps_masker = NiftiMapsMasker(maps_img=nb.load(difumo_path))
    maps_masker.fit()

    print("Evaluating...")
    evaluate(
        dataloaders,
        encodec_model,
        vae_model,
        args.target_run_name,
        label2task,
        maps_masker,
        max_seq_len=args.max_seq_len,
        target_label=args.target_label,
        device=device,
    )
    print("finish :", args.classifier_model_path)
