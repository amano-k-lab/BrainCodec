import argparse
import typing as tp

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import torch
from nilearn import plotting
from nilearn.datasets import fetch_atlas_difumo
from nilearn.image import iter_img
from nilearn.maskers import NiftiMapsMasker

from audiocraft.data.fMRI_dataset import SegmentInfo
from audiocraft.models.RawCSM import RawCSM
from audiocraft.solvers import builders


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
    for i, info in enumerate(infos):
        _tr = info.tr
        raw_trs.append(_tr)
        fMRI_duration = info.duration
        bold_durations.append(fMRI_duration)
        _tr_tensor = torch.arange(0, fMRI_duration, device=device) * _tr

        # fMRI_duration が x.size(-1) より大きい場合は、テンソルをクロップ
        if fMRI_duration > x.size(-1):
            _tr_tensor = _tr_tensor[: x.size(-1)]
        else:
            # x.size(-1) 未満の場合は、残りの部分をゼロで埋める
            pad_length = x.size(-1) - fMRI_duration
            pad_id = info.max_tr + info.tr_precision
            _tr_tensor = torch.cat((_tr_tensor, pad_id * torch.ones(pad_length, device=device)))

        # ID にする
        _tr_tensor = (_tr_tensor / info.tr_precision).floor()

        # 結果を tr テンソルに追加
        tr[i] = _tr_tensor

    if return_raw_tr is True:
        return x, tr, raw_trs, bold_durations
    return x, tr


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
    return rec


def fMRI_write(rec_bold, output_path, tr, block_num=6, start_time=0):
    rec_bold = rec_bold.cpu().numpy().T

    time_slice = np.arange(start_time, start_time + block_num).astype(int)

    rec_bold = rec_bold[time_slice]

    reconstructed_img = maps_masker.inverse_transform(rec_bold)

    fig, axes = plt.subplots(figsize=(block_num * 3, block_num * 2), ncols=2, nrows=block_num // 2)
    axes = axes.reshape(-1)
    for i, (step, cmap) in enumerate(zip(time_slice, iter_img(reconstructed_img))):
        plotting.plot_stat_map(
            cmap,
            figure=fig,
            axes=axes[i],
            colorbar=False,
            title=f"t={step*tr}",
            cut_coords=(0, -22, 16),
        )

    # plt.suptitle(output_path.stem, fontsize=64, fontweight="bold")
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csm_paths", type=str, nargs="+", help="List of paths of checkpoint", required=True)
    args = parser.parse_args()
    use_layer = [0, 1, 2, 3, 4, 5, 6, 7]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    classifier_models = [RawCSM.get_pretrained(p) for p in args.csm_paths]
    cfg = classifier_models[0].lm.cfg
    cfg["use_layer"] = use_layer
    cfg["execute_only"] = "evaluate"

    print("Loading datasets...")
    dataloaders = builders.get_fMRI_datasets(cfg)

    print("Preparing maps masker...")
    difumo_path = fetch_atlas_difumo(dimension=cfg.space_dim, resolution_mm=2, legacy_format=False).maps
    maps_masker = NiftiMapsMasker(maps_img=nb.load(difumo_path))
    maps_masker.fit()

    print("Evaluating...")
    batch = next(iter(dataloaders["evaluate"]))
    bold_gt, tr, raw_tr, bold_durations = _preprocess_tr(batch, device, return_raw_tr=True)  # (B, dim, time)
    target_duration = bold_durations[0]
    target_tr = raw_tr[0]
    tr = tr[0].unsqueeze(0)[:, :target_duration]
    bold_gt = bold_gt[0].unsqueeze(0)[:, :, :target_duration]
    print("bold gt: ", bold_gt.shape)
    for i, csm in enumerate(classifier_models):
        if csm.compression_model is None:
            rec_bold = bold_gt
        else:
            rec_bold = get_reconstructed_fMRI(csm.compression_model, cfg, bold_gt, tr)
        prompt_bold = rec_bold[:, :, : target_duration // 2]
        with csm.autocast:
            gen_bold = csm.lm.generate(
                prompt_bold,
                [],
                [target_tr],
                max_gen_len=target_duration,
            )
        print(f"gen bold {i}: ", gen_bold.shape)
        fMRI_write(
            rec_bold[0][:, :target_duration],
            f"reconstructed_{i}.png",
            target_tr,
            start_time=target_duration // 2 - 2,
        )
        fMRI_write(
            gen_bold[0][:, :target_duration],
            f"generated_{i}.png",
            target_tr,
            start_time=target_duration // 2 - 2,
        )
