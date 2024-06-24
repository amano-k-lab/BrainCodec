# evaluate 用のデータセットを読み込んで，正しく予測できたデータの内，一番自信もって予測できた bold を画像として出力する．
# layer ごとにも出力する.
import argparse
import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nb
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
    block_num = time_step  # 全部の時刻を表示

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
def get_reconstructed_fMRI(codec_model, cfg, bold: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    output = codec_model.encode(bold, tr)
    if isinstance(output, torch.Tensor):
        rec = codec_model.decode(output, tr, noise_disable=True, use_layer=cfg.use_layer)
    elif len(output) == 2:
        codes, logvar = output
        rec = codec_model.decode(codes, tr, logvar, noise_disable=True, use_layer=cfg.use_layer)
    elif len(output) == 3:
        codes, mu, logvar = output
        rec = codec_model.decode(codes, tr, mu, logvar, noise_disable=True, use_layer=cfg.use_layer)
    else:
        raise RuntimeError("Only EncodecfMRIModel model is supported.")
    return rec, codes


def evaluate(model, vae_model, dataloaders, cfg, device, label2task):
    loader = dataloaders["evaluate"]
    metrics = {}

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        bold, tr, raw_trs, bold_durations, labels, names = _preprocess_tr(batch, device, return_raw_tr=True)
        codec_model = model.compression_model
        rec_bold, codes = get_reconstructed_fMRI(codec_model, cfg, bold, tr)
        rec_vae_bold, _ = get_reconstructed_fMRI(vae_model, cfg, bold, tr)
        with torch.no_grad():
            y_pred = model.get_prediction(rec_bold, bold_durations, raw_trs)

        y = labels.cpu()  # should already be on CPU but just in case
        y_pred = y_pred.cpu()  # should already be on CPU but just in case
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).float()

        for idx, _ in enumerate(rec_bold):
            c = correct[idx]
            t = y[idx]
            pred_values = y_pred[idx]
            bold_duration = bold_durations[idx]
            _tr = raw_trs[idx]
            if c.item() == 0:
                continue
            task = label2task[t.item()]
            p = pred_values[t.item()].item()
            info = {
                "gt_bold": bold[idx],
                "rec_bold": rec_bold[idx],
                "rec_vae_bold": rec_vae_bold[idx],
                "code": codes[idx] if codes is not None else None,
                "p": p,
                "bold_duration": bold_duration,
                "tr": _tr,
                "name": names[idx],
            }
            if bold_duration > 50:
                continue
            if task not in metrics:
                metrics[task] = {}
                metrics[task][f"label_{t.item()}"] = [info]
            else:
                if f"label_{t.item()}" not in metrics[task]:
                    metrics[task][f"label_{t.item()}"] = [info]
                else:
                    if metrics[task][f"label_{t.item()}"][-1]["p"] < p:
                        metrics[task][f"label_{t.item()}"].append(info)
                    elif (metrics[task][f"label_{t.item()}"][-1]["p"] < p + 1e-8) and (
                        metrics[task][f"label_{t.item()}"][-1]["bold_duration"] < bold_duration
                    ):
                        metrics[task][f"label_{t.item()}"].append(info)
    return metrics


if __name__ == "__main__":
    # This code is only for CSM model
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_model_path", type=str, help="Path to classifier model checkpoint", required=True)
    parser.add_argument("--vae_codec_model_path", type=str, help="Path to VAE model checkpoint", required=True)
    parser.add_argument("--output_dir", type=str, help="Output directory", required=True)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading model...")
    classifier_model = DownstreamRawCSM.get_pretrained(args.classifier_model_path)
    encodec_model = classifier_model.compression_model
    vae_model = CompressionfMRISolver.model_from_checkpoint(args.vae_codec_model_path).to(device)
    cfg = classifier_model.lm.cfg
    cfg["execute_only"] = "evaluate"
    print("Loading datasets...")
    dataloaders = builders.get_fMRI_datasets(cfg)

    print("Preparing maps masker...")
    difumo_path = fetch_atlas_difumo(dimension=cfg.space_dim, resolution_mm=2, legacy_format=False).maps
    maps_masker = NiftiMapsMasker(maps_img=nb.load(difumo_path))
    maps_masker.fit()

    print("Evaluating...")
    metrics = evaluate(classifier_model, vae_model, dataloaders, cfg, device, label2task)
    print("output images...")
    each_probs = {}
    total_count = 0
    for task, data in metrics.items():
        for label, infos in data.items():
            for rank, info in enumerate(infos[-3:]):
                total_count += 1
    progress_bar = tqdm(total=total_count)
    for task, data in metrics.items():
        for label, infos in data.items():
            for rank, info in enumerate(infos[-3:]):
                _tr = info["tr"]
                # gt output
                output_path = (
                    output_dir
                    / f"Task_{task}_{label}_{rank}_{info['name']}_gt_p_{round(info['p'], 3)}_tr_{_tr:.2f}.png"
                )
                fMRI_write(
                    info["gt_bold"][:, : info["bold_duration"]],
                    output_path,
                    maps_masker,
                    _tr,
                    task,
                )
                # rec output
                output_path = (
                    output_dir
                    / f"Task_{task}_{label}_{rank}_{info['name']}_rec_p_{round(info['p'], 3)}_tr_{_tr:.2f}.png"
                )
                fMRI_write(
                    info["rec_bold"][:, : info["bold_duration"]],
                    output_path,
                    maps_masker,
                    _tr,
                    task,
                )
                # rec by vae
                output_path = (
                    output_dir
                    / f"Task_{task}_{label}_{rank}_{info['name']}_rec_vae_p_{round(info['p'], 3)}_tr_{_tr:.2f}.png"
                )
                fMRI_write(
                    info["rec_vae_bold"][:, : info["bold_duration"]],
                    output_path,
                    maps_masker,
                    _tr,
                    task,
                )
                each_probs[task] = {}
                if info["code"] is not None:
                    # each layer's rec
                    tr_tensor = tr_preprocess(
                        cfg.dataset.max_tr,
                        cfg.dataset.tr_precision,
                        [info["tr"]],
                        50,
                        [info["bold_duration"]],
                    ).to(device)
                    for i in range(8):
                        if i == 0:
                            use_layer = [i]
                            out = encodec_model.decode(
                                info["code"].unsqueeze(0),
                                tr_tensor,
                                None,
                                None,
                                use_layer=use_layer,
                            ).detach()
                            # get prob
                            y_pred = classifier_model.get_prediction(out, [info["bold_duration"]], [info["tr"]])
                            _label = int(label.split("_")[-1])
                            p = y_pred[0, _label].item()
                            each_probs[task][f"layer_{i}"] = str(round(p, 3))
                            output_path = (
                                output_dir
                                / f"Task_{task}_{label}_{rank}_{info['name']}_rec_layer_{i}_p_{round(p, 3)}_tr_{_tr:.2f}.png"
                            )
                            fMRI_write(
                                out[0][:, : info["bold_duration"]],
                                output_path,
                                maps_masker,
                                _tr,
                                task,
                            )
                        elif i == 3:
                            use_layer = list(range(i + 1))
                            out = encodec_model.decode(
                                info["code"].unsqueeze(0),
                                tr_tensor,
                                None,
                                None,
                                use_layer=use_layer,
                            ).detach()
                            # get prob
                            y_pred = classifier_model.get_prediction(out, [info["bold_duration"]], [info["tr"]])
                            p = y_pred[0, _label].item()
                            each_probs[task][f"layer_0to{i}"] = str(round(p, 3))
                            output_path = (
                                output_dir
                                / f"Task_{task}_{label}_{rank}_{info['name']}_rec_layer_0to{i}_p_{round(p, 3)}_tr_{_tr:.2f}.png"
                            )
                            fMRI_write(
                                out[0][:, : info["bold_duration"]],
                                output_path,
                                maps_masker,
                                _tr,
                                task,
                            )
                progress_bar.update(1)

    print("finish :", args.classifier_model_path)
