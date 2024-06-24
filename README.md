# BrainCraft

BrainCraft は AudioCraft をベースとした pytorch ライブラリです．
BrainCraft は現在 BrainCodec の推論・訓練コードを含んでいます．


## Installation
BrainCraft requires Python 3.9, PyTorch 2.1.0. To install BrainCraft, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
pip install 'torch>=2.1'
pip install -e .
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Models
At the moment, BrainCraft contains the training code and inference code for:

* [BrainCodec](docs/BRAINCODEC.md): A state-of-the-art neural codec model for fMRI data

## License
* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).

## Citation

For the general framework of BrainCraft, please cite the following.
```
```

## Acknowledgement
このリポジトリは [AudioCraft](https://github.com/facebookresearch/audiocraft) を元に実装されています．
