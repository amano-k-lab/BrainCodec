import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from tqdm import tqdm

from audiocraft.solvers import CompressionfMRISolver


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=8e-9, maxiters=1000000, logspace=False):
    # https://github.com/anucvml/ddn/blob/master/ddn/pytorch/optimal_transport.py
    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))

    for i in tqdm(range(maxiters), desc="Sinkhorn"):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        elif i % (maxiters // 10) == 0:
            print("Current error:", torch.max(torch.abs(beta - c)))
        P = P / beta.view(B, 1, W) * c

    return P


def wasserstein_distance_pytorch(task_codebooks, rest_codebooks, device):
    task_codebooks = task_codebooks.to(device)
    rest_codebooks = rest_codebooks.to(device)
    # NOTE: l2 normalization とかすると，正しい比較にならない気がした (大きさも大事なので)
    # calc cost matrix
    cost = torch.cdist(task_codebooks, rest_codebooks, p=2)
    # min-max normalization
    # NOTE: l2 normalization の代わりに，0~1 に値を収める．これは層間での比較のため
    wasserstein_distance = []
    for i in range(cost.shape[0]):
        cost[i] = (cost[i] - torch.min(cost[i])) / (torch.max(cost[i]) - torch.min(cost[i]))
    _ot = sinkhorn(cost)
    # calc wasserstein distance
    wasserstein_distance = torch.sum(_ot * cost, dim=(1, 2))
    return wasserstein_distance.cpu().numpy()


def wasserstein_distance_numpy(
    task_codebooks,
    rest_codebooks,
    device,
    data_normalize=False,
    data_standardize=True,
    cost_normalize=False,
    cost_standardize=False,
):
    task_codebooks = task_codebooks.to(device)
    rest_codebooks = rest_codebooks.to(device)
    # normalize
    if data_normalize is True:
        raise NotImplementedError
    if data_standardize is True:
        task_codebooks = (task_codebooks - torch.mean(task_codebooks, dim=1, keepdim=True)) / (
            torch.std(task_codebooks, dim=1, keepdim=True) + 1e-9
        )
        rest_codebooks = (rest_codebooks - torch.mean(rest_codebooks, dim=1, keepdim=True)) / (
            torch.std(rest_codebooks, dim=1, keepdim=True) + 1e-9
        )
    # calc cost matrix
    cost = torch.cdist(task_codebooks, rest_codebooks, p=2).cpu().numpy()
    # min-max normalization
    wasserstein_distance = []
    for i in tqdm(range(cost.shape[0]), desc="Wasserstein"):
        if cost_normalize is True:
            cost[i] = (cost[i] - np.min(cost[i])) / (np.max(cost[i]) - np.min(cost[i]))
        if cost_standardize is True:
            cost[i] = (cost[i] - np.mean(cost[i])) / (np.std(cost[i]) + 1e-9)
        n = cost.shape[1]
        a, b = np.ones((n,)) / n, np.ones((n,)) / n
        _ot = ot.emd(a, b, cost[i])
        wasserstein_distance.append(np.sum(_ot * cost[i]))
    return np.array(wasserstein_distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-fmri-encodec-path", type=str, default=None)
    parser.add_argument("--rest-fmri-encodec-path", type=str, default=None)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--exp-name", type=str)
    args = parser.parse_args()

    codebook_shape = (8, 1024, 128)
    bins = 1

    args.output_path.mkdir(exist_ok=True, parents=True)

    if not (args.output_path / f"wasserstain_distance_{args.exp_name}.npy").exists():
        if args.task_fmri_encodec_path is not None:
            print("Loading task models...")
            task_model = CompressionfMRISolver.model_from_checkpoint(args.task_fmri_encodec_path)
            task_codebooks = torch.stack([vq_layer._codebook.embed for vq_layer in task_model.quantizer.vq.layers])
        else:
            task_codebooks = torch.randn(codebook_shape)
        if args.rest_fmri_encodec_path is not None:
            print("Loading rest models...")
            rest_model = CompressionfMRISolver.model_from_checkpoint(args.rest_fmri_encodec_path)
            rest_codebooks = torch.stack([vq_layer._codebook.embed for vq_layer in rest_model.quantizer.vq.layers])
        else:
            rest_codebooks = torch.randn(codebook_shape)
        assert task_codebooks.shape == rest_codebooks.shape, (
            task_codebooks.shape,
            rest_codebooks.shape,
        )

        device = "cpu"
        print("device:", device)
        print("Calculating wasserstein distance...")
        wasserstain_distance = wasserstein_distance_numpy(task_codebooks, rest_codebooks, device)
        np.save(args.output_path / f"wasserstain_distance_{args.exp_name}.npy", wasserstain_distance)
    else:
        wasserstain_distance = np.load(args.output_path / f"wasserstain_distance_{args.exp_name}.npy")

    print("Plotting...")
    plt.figure(figsize=(12, 8))
    plt.plot(wasserstain_distance)
    plt.title(args.exp_name)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.xticks(range(0, codebook_shape[0], bins), range(0, codebook_shape[0], bins))  # Show only every 4th index
    plt.savefig(args.output_path / f"plot_{args.exp_name}.png")
    print("Done.")
