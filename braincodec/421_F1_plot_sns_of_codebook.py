import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap

from audiocraft.solvers import CompressionfMRISolver

fig, axes = plt.subplots(1, 2, figsize=(15, 12))

parser = argparse.ArgumentParser()
parser.add_argument("--task_codec_path", type=str, help="Path to task codec checkpoint")
parser.add_argument("--rest_codec_path", type=str, help="Path to rest codec checkpoint")
args = parser.parse_args()
n_q = 8

for code_book in range(n_q):

    def get_codebooks_from_checkpoint(checkpoint_path):
        task_model = CompressionfMRISolver.model_from_checkpoint(checkpoint_path)
        codebooks = torch.stack([vq_layer._codebook.embed for vq_layer in task_model.quantizer.vq.layers])
        return codebooks[code_book]

    codebooks_1 = get_codebooks_from_checkpoint(args.task_codec_path)
    codebooks_2 = get_codebooks_from_checkpoint(args.rest_codec_path)
    codebooks_3 = torch.randn_like(codebooks_1)

    codebooks_combined = torch.cat((codebooks_1, codebooks_2, codebooks_3), dim=0).cpu().numpy()

    reducer = umap.UMAP()
    reduced_codebooks = reducer.fit_transform(codebooks_combined)

    labels = ["Task"] * codebooks_1.shape[0] + ["Rest"] * codebooks_2.shape[0] + ["Random"] * codebooks_3.shape[0]

    df = pd.DataFrame(reduced_codebooks, columns=["UMAP1", "UMAP2"])
    df["Label"] = labels

    sns.scatterplot(ax=axes[code_book], data=df, x="UMAP1", y="UMAP2", hue="Label", palette="Set1")
    axes[code_book].set_xlabel("UMAP1", fontsize=24)
    axes[code_book].set_ylabel("UMAP2", fontsize=24)
    if code_book == 0:
        axes[code_book].legend(fontsize="xx-large", loc="upper left")
    else:
        axes[code_book].legend_.remove()

plt.tight_layout()
plt.savefig(f"umap_codebooks_{code_book}.png")
