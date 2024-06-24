import argparse
import multiprocessing
import typing as tp

import flashy
import omegaconf
import torch
from tqdm import tqdm

from audiocraft.data.fMRI_dataset import SegmentInfo
from audiocraft.models.DownstreamRawCSM import DownstreamRawCSM
from audiocraft.solvers import builders
from audiocraft.solvers.compression_fMRI import CompressionfMRISolver
from audiocraft.utils.utils import get_pool_executor


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
    for i, info in enumerate(infos):
        _tr = info.tr
        labels.append(info.meta.label)
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

    labels = torch.tensor(labels, device=device).long()

    if return_raw_tr is True:
        return x, tr, raw_trs, bold_durations, labels
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
    return rec


def evaluate(model, codec_model, dataloaders, cfg, device, label2task):
    loader = dataloaders["evaluate"]
    average = flashy.averager()

    pendings = []
    ctx = multiprocessing.get_context("spawn")
    with get_pool_executor(cfg.evaluate.num_workers, mp_context=ctx) as pool:
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            x, tr, raw_trs, bold_durations, labels = _preprocess_tr(batch, device, return_raw_tr=True)
            if codec_model is not None:
                x = get_reconstructed_fMRI(codec_model, cfg, x, tr)
            with torch.no_grad():
                y_pred = model.get_prediction(x, bold_durations, raw_trs)

            y = labels.cpu()  # should already be on CPU but just in case
            y_pred = y_pred.cpu()  # should already be on CPU but just in case
            pendings.append(pool.submit(evaluate_fMRI_acc, y_pred, y, cfg, label2task))  # Need to change this

        for pending in pendings:
            metrics = pending.result()
            metrics = average(metrics)

    metrics = flashy.distrib.average_metrics(metrics, len(loader))
    return metrics


def evaluate_fMRI_acc(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    cfg: omegaconf.DictConfig,
    label2task: tp.Dict[int, str],
) -> dict:
    metrics = {}
    # 全体の acc
    accuracy_score = builders.get_loss("accuracy", cfg)
    metrics["accuracy"] = accuracy_score(y_pred, y)
    # タスク別 acc
    predicted = torch.argmax(y_pred, dim=1)
    correct = (predicted == y).float()
    task_lst = []
    for c, t in zip(correct, y):
        task = "accuracy_" + label2task[t.item()]
        if task not in metrics:
            metrics[task] = []
            task_lst.append(task)
        metrics[task].append(c.item())
    for task in task_lst:
        metrics[task] = torch.tensor(metrics[task]).mean()
    f1_score = builders.get_loss("f1", cfg)
    f1 = f1_score(y_pred, y)
    metrics["f1"] = f1

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-model-path", type=str, help="Path to classifier model checkpoint", required=True)
    parser.add_argument("--use-layer", type=int, nargs="+", help="List of layer to use", required=True)
    parser.add_argument(
        "--rest_codec_model_path", type=str, help="Path to resting-state fMRI codec model checkpoint", default=None
    )
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

    print("Loading model...")
    classifier_model = DownstreamRawCSM.get_pretrained(args.classifier_model_path)
    encodec_model = classifier_model.compression_model
    if args.rest_codec_model_path is not None:
        rest_encodec_model = CompressionfMRISolver.model_from_checkpoint(
            "/home/andante/workspace/proj_brain_multimodal/logs/audiocraft/xps/9ba91ca7/checkpoint_200.th"
        ).to("cuda")
        for vq_layer, rest_vq_layer in zip(encodec_model.quantizer.vq.layers, rest_encodec_model.quantizer.vq.layers):
            vq_layer._codebook.embed = rest_vq_layer._codebook.embed
    # encodec_model = None
    cfg = classifier_model.lm.cfg
    cfg["use_layer"] = args.use_layer
    cfg["execute_only"] = "evaluate"
    print("Loading datasets...")
    dataloaders = builders.get_fMRI_datasets(cfg)

    print("Evaluating...")
    metrics = evaluate(classifier_model, encodec_model, dataloaders, cfg, device, label2task)
    for k, v in sorted(metrics.items()):
        print(f"{k}: {round(v, 3)}")
    print("finish :", args.use_layer)
