# BrainCodec: Neural fMRI codec for the decoding of cognitive brain states

## preprocessing:
### dataset
* upstream, downstream 用のデータを [learning-from-brains](https://github.com/athms/learning-from-brains) からダウンロードする
* resting-state fMRI に関しては，[こちら](https://drive.google.com/drive/folders/1RYetEquBbOsWz5P3Z5i0levYhIev3a92?usp=sharing) からダウンロードされたい．
  * 但し，論文にあるように SRPBS Traveling Subject MRI Dataset についてはライセンス上再配布ができないため，自身でダウンロードし，tarfile にしてください．
* 自分で tarfile を作成する手順は以下の通りです:
  * 次のコードを利用して前処理を行ったのち，`learning-from-brains/scripts/dataprep/upstream/dataprep.py` を利用して tarfile を作成する．
```
sudo docker run -ti --rm \
    -v ${/path/to/input_data}:/data:ro \
    -v ${/path/to/output_data}:/out \
    -v ${/path/to/freesurfer_license}:/opt/freesurfer/license.txt \
    nipreps/fmriprep:20.2.3 \
    /data /out/out \
    participant \
    --skip_bids_validation \
    --output-spaces MNI152NLin6Asym:res-2 MNI152NLin2009cAsym:res-2 fsaverage:den-10k T1w func \
    --random-seed 0 \
    --skull-strip-fixed-seed \
    --dummy-scans 0 \
    --fd-spike-threshold 0.5 \
    --participant-label sub-039 \
    --nprocs 2
```
### split 分割
* upstream 用の split を作成する:
```
python audiocraft/data/devide_downstream_data.py  \
    --data /path/to/dataset --output /path/to/output
```
* downstream 用の split を作成する:
```
# For HCP
python audiocraft/data/devide_downstream_data.py  \
    --data /path/to/dataset --output /path/to/output
# For MDTB
python audiocraft/data/devide_downstream_data.py  \
    --data /path/to/dataset --output /path/to/output  \
    --n-val-subjects-per-dataset 3 --n-test-subjects-per-dataset 9 --label_name task_label.pyd
```
### egs 作成
* 各 upstream, downstream で train/valid/test 用をそれぞれ作成する:
```
python -m audiocraft.data.fMRI_dataset \
    /path/to/dataset \
    /path/to/output \
    --complete --resolve
# 具体例:
python -m audiocraft.data.fMRI_dataset \
    data/downstream_split_hcp_n-48/train \
    egs/fMRI_downstream/HCP_n-48/train/data.jsonl.gz \
    --complete --resolve
```
* 再現性のために，筆者が利用した egs は[こちら](https://drive.google.com/drive/folders/1nDGPngv7atDb-ZQKfsPwWLQHtNVKPCrr?usp=sharing) で配布する．記載されている絶対パスを自分の環境に合うように変更すれば利用可能．

## training:
* config/teams 以下に yaml を作成する．default.yaml を複製し，dora_dir と reference_dir を好きなパスに指定する．
* 訓練を実行する:
```
AUDIOCRAFT_TEAM="作成した teams の yaml 名"
EXP_NAME="任意の実験名"
SOLVER="使用する solver"
ARGS="上書きしたいオプションを指定"
export AUDIOCRAFT_TEAM=${AUDIOCRAFT_TEAM}
dora run solver=${SOLVER} ${ARGS} wandb.name=${EXP_NAME}

# 具体例:
EXP_NAME=braincodec_default
SOLVER=compression_fMRI/braincodec_base
ARGS=""
```
* すべての config は論文で最終的に利用されたオプションに設定してある．CSM や linear モデルの訓練時には作成した braincodec の checkpoint を指定する必要があることに注意．

## inference:
* 以下のコードでモデルを読み込み, forward や encode, decode を行う:
```
from audiocraft.solvers.compression_fMRI import CompressionfMRISolver

model, cfg = CompressionSolver.model_from_checkpoint(path_to_checkpoint, need_cfg=True)
```
* 具体的な fMRI データの再構成の例などは [422_F3_plot_evaluate_fMRI_data](../braincodec/422_F3_plot_evaluate_fMRI_data.py) を参照


## Appendix
### F2
シンクホーンアルゴリズムを実施するコード
encodec path に checkpoint へのパスを指定する．
指定しないと自動でガウスノイズからの plot になる．
`compare_two_encodec.py` によって計算したのち，`plot_all_data.py` において比較したいデータを指定し描画する．
