# BrainCodec: Neural fMRI Codec for the Decoding of Cognitive Brain States

## Preprocessing:

### Dataset
* Download the data for upstream and downstream tasks from [learning-from-brains](https://github.com/athms/learning-from-brains).
* For resting-state fMRI, download from [here](https://drive.google.com/drive/folders/1RYetEquBbOsWz5P3Z5i0levYhIev3a92?usp=sharing).
  * Note: Due to licensing restrictions, the SRPBS Traveling Subject MRI Dataset mentioned in the paper cannot be redistributed. You need to download it yourself and create a tarfile.
* Follow the steps below to create your own tarfile:
  * Perform preprocessing using the following code, and then create a tarfile using `learning-from-brains/scripts/dataprep/upstream/dataprep.py`.
  ```bash
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

### Split Creation
* Create splits for upstream tasks:
  ```bash
  python audiocraft/data/devide_downstream_data.py  \
      --data /path/to/dataset --output /path/to/output
  ```

* Create splits for downstream tasks:
  ```bash
  # For HCP
  python audiocraft/data/devide_downstream_data.py  \
      --data /path/to/dataset --output /path/to/output
  # For MDTB
  python audiocraft/data/devide_downstream_data.py  \
      --data /path/to/dataset --output /path/to/output  \
      --n-val-subjects-per-dataset 3 --n-test-subjects-per-dataset 9 --label_name task_label.pyd
  ```

### Creating egs
* Create train/valid/test sets for each upstream and downstream task:
  ```bash
  python -m audiocraft.data.fMRI_dataset \
      /path/to/dataset \
      /path/to/output \
      --complete --resolve
  # Example:
  python -m audiocraft.data.fMRI_dataset \
      data/downstream_split_hcp/train \
      egs/fMRI_downstream/HCP/train/data.jsonl.gz \
      --complete --resolve
  ```

* For reproducibility, the egs used by the authors are distributed [here](https://drive.google.com/drive/folders/1nDGPngv7atDb-ZQKfsPwWLQHtNVKPCrr?usp=sharing). Adjust the absolute paths to suit your environment.

## Training:
* Create a YAML file under config/teams. Duplicate default.yaml and set dora_dir and reference_dir to your preferred paths.
* Run the training:
  ```bash
  AUDIOCRAFT_TEAM="Name of the YAML file created in teams"
  EXP_NAME="Arbitrary experiment name"
  SOLVER="Solver to be used"
  ARGS="Options to overwrite"
  export AUDIOCRAFT_TEAM=${AUDIOCRAFT_TEAM}
  dora run -d solver=${SOLVER} ${ARGS} wandb.name=${EXP_NAME}

  # Example:
  EXP_NAME=braincodec_default
  SOLVER=compression_fMRI/braincodec_base
  ARGS=""
  ```

* All configs are set to the options ultimately used in the paper. Note that when training CSM or linear models, you need to specify the checkpoint of the created BrainCodec.

## Inference:
* Load the model and perform forward, encode, and decode operations with the following code:
  ```python
  from audiocraft.solvers.compression_fMRI import CompressionfMRISolver

  model, cfg = CompressionSolver.model_from_checkpoint(path_to_checkpoint, need_cfg=True)
  ```

* For specific examples of fMRI data reconstruction, refer to [422_F3_plot_evaluate_fMRI_data](../braincodec/422_F3_plot_evaluate_fMRI_data.py).

## Appendix
### 4.2.1 and F.1: UMAP Plot of Codebook
UMAP plots of the codebook acquired through the training of BrainCodec. You can visualize this by specifying the BrainCodec checkpoints trained on task fMRI and resting-state fMRI in `braincodec/421_F1_plot_sns_of_codebook.py`.

### 4.2.2: Performance Changes with Varying Layer Numbers
Code to check performance changes on the HCP dataset when the number of layers is varied. You can output results by specifying the CSM checkpoint trained on HCP data and the layer number to use in `braincodec/422_calc_acc_for_each_layer_with_CSM.py`. By specifying the `rest_codec_model_path` option, you can replace the codebook of the codec model with only the codebook specified by BrainCodec.

### 4.2.2 and F3: Visualization and Evaluation of Codec Model Reconstruction
By specifying the CSM checkpoint and the path to BrainVAE in `braincodec/422_F3_plot_evaluate_fMRI_data.py`, you can visualize the reconstructed data by BrainCodec and BrainVAE along with the Ground Truth data.

In `braincodec/422_F3_create_mean_fMRI_data.py`, you can create an average image for a specific task by specifying the target task label and the trial name to use for reconstruction, and then calculate the L1 distance between the codec model's reconstructed fMRI and the average task image.

### D2: fMRI Generation with CSM
In `braincodec/D2_rawcsm_generate.py`, you can generate the continuation of a given fMRI data conditioned on the CSM checkpoint.

F.2: Codebook Distance Using Sinkhorn Algorithm
Code to implement the Sinkhorn algorithm. Specify the path to the checkpoint for the Encodec path. If not specified, it defaults to a plot from Gaussian noise. After calculating with `braincodec/F2_sinkhorn_algorithm/compare_two_encodec.py`, specify the data you want to compare and plot it with `braincodec/F2_sinkhorn_algorithm/plot_all_data.py`.
