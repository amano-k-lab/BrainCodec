# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
API that can manage the storage and retrieval of generated samples produced by experiments.

It offers the following benefits:
* Samples are stored in a consistent way across epoch
* Metadata about the samples can be stored and retrieved
* Can retrieve audio
* Identifiers are reliable and deterministic for prompted and conditioned samples
* Can request the samples for multiple XPs, grouped by sample identifier
* For no-input samples (not prompt and no conditions), samples across XPs are matched
  by sorting their identifiers
"""

import hashlib
import json
import logging
import re
import time
import typing as tp
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import dora
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import torch
import webdataset as wds
from nibabel import Nifti1Image
from nilearn import plotting
from nilearn.datasets import fetch_atlas_difumo
from nilearn.image import iter_img, math_img
from nilearn.maskers import NiftiMapsMasker

from ...data.audio import audio_read, audio_write
from ...data.fMRI_dataset import fMRI_read

logger = logging.getLogger(__name__)


@dataclass
class ReferenceSample:
    id: str
    path: str
    duration: float


@dataclass
class ReferencefMRISample:
    id: str
    path: str
    duration: int
    tr: float


@dataclass
class Sample:
    id: str
    path: str
    epoch: int
    duration: float
    conditioning: tp.Optional[tp.Dict[str, tp.Any]]
    prompt: tp.Optional[ReferenceSample]
    reference: tp.Optional[ReferenceSample]
    generation_args: tp.Optional[tp.Dict[str, tp.Any]]

    def __hash__(self):
        return hash(self.id)

    def audio(self) -> tp.Tuple[torch.Tensor, int]:
        return audio_read(self.path)

    def audio_prompt(self) -> tp.Optional[tp.Tuple[torch.Tensor, int]]:
        return audio_read(self.prompt.path) if self.prompt is not None else None

    def audio_reference(self) -> tp.Optional[tp.Tuple[torch.Tensor, int]]:
        return audio_read(self.reference.path) if self.reference is not None else None


@dataclass
class SamplefMRI:
    id: str
    path: str
    epoch: int
    duration: int
    tr: float
    conditioning: tp.Optional[tp.Dict[str, tp.Any]]
    prompt: tp.Optional[ReferencefMRISample]
    reference: tp.Optional[ReferencefMRISample]
    generation_args: tp.Optional[tp.Dict[str, tp.Any]]

    def __hash__(self):
        return hash(self.id)

    def fMRI(self) -> tp.Tuple[torch.Tensor, float]:
        return fMRI_read(self.path)

    def fMRI_prompt(self) -> tp.Optional[tp.Tuple[torch.Tensor, float]]:
        return fMRI_read(self.prompt.path) if self.prompt is not None else None

    def fMRI_reference(self) -> tp.Optional[tp.Tuple[torch.Tensor, float]]:
        return fMRI_read(self.reference.path) if self.reference is not None else None


class SampleManager:
    """Audio samples IO handling within a given dora xp.

    The sample manager handles the dumping and loading logic for generated and
    references samples across epochs for a given xp, providing a simple API to
    store, retrieve and compare audio samples.

    Args:
        xp (dora.XP): Dora experiment object. The XP contains information on the XP folder
            where all outputs are stored and the configuration of the experiment,
            which is useful to retrieve audio-related parameters.
        map_reference_to_sample_id (bool): Whether to use the sample_id for all reference samples
            instead of generating a dedicated hash id. This is useful to allow easier comparison
            with ground truth sample from the files directly without having to read the JSON metadata
            to do the mapping (at the cost of potentially dumping duplicate prompts/references
            depending on the task).
    """

    def __init__(self, xp: dora.XP, map_reference_to_sample_id: bool = False):
        self.xp = xp
        self.base_folder: Path = xp.folder / xp.cfg.generate.path
        self.reference_folder = self.base_folder / "reference"
        self.map_reference_to_sample_id = map_reference_to_sample_id
        self.samples: tp.List[Sample] = []
        self._load_samples()

    @property
    def latest_epoch(self):
        """Latest epoch across all samples."""
        return max(self.samples, key=lambda x: x.epoch).epoch if self.samples else 0

    def _load_samples(self):
        """Scan the sample folder and load existing samples."""
        jsons = self.base_folder.glob("**/*.json")
        with ThreadPoolExecutor(6) as pool:
            self.samples = list(pool.map(self._load_sample, jsons))

    @staticmethod
    @lru_cache(2**26)
    def _load_sample(json_file: Path) -> Sample:
        with open(json_file, "r") as f:
            data: tp.Dict[str, tp.Any] = json.load(f)
        # fetch prompt data
        prompt_data = data.get("prompt")
        prompt = (
            ReferenceSample(
                id=prompt_data["id"],
                path=prompt_data["path"],
                duration=prompt_data["duration"],
            )
            if prompt_data
            else None
        )
        # fetch reference data
        reference_data = data.get("reference")
        reference = (
            ReferenceSample(
                id=reference_data["id"],
                path=reference_data["path"],
                duration=reference_data["duration"],
            )
            if reference_data
            else None
        )
        # build sample object
        return Sample(
            id=data["id"],
            path=data["path"],
            epoch=data["epoch"],
            duration=data["duration"],
            prompt=prompt,
            conditioning=data.get("conditioning"),
            reference=reference,
            generation_args=data.get("generation_args"),
        )

    def _init_hash(self):
        return hashlib.sha1()

    def _get_tensor_id(self, tensor: torch.Tensor) -> str:
        hash_id = self._init_hash()
        hash_id.update(tensor.numpy().data)
        return hash_id.hexdigest()

    def _get_sample_id(
        self,
        index: int,
        prompt_wav: tp.Optional[torch.Tensor],
        conditions: tp.Optional[tp.Dict[str, str]],
    ) -> str:
        """Computes an id for a sample given its input data.
        This id is deterministic if prompt and/or conditions are provided by using a sha1 hash on the input.
        Otherwise, a random id of the form "noinput_{uuid4().hex}" is returned.

        Args:
            index (int): Batch index, Helpful to differentiate samples from the same batch.
            prompt_wav (torch.Tensor): Prompt used during generation.
            conditions (dict[str, str]): Conditioning used during generation.
        """
        # For totally unconditioned generations we will just use a random UUID.
        # The function get_samples_for_xps will do a simple ordered match with a custom key.
        if prompt_wav is None and not conditions:
            return f"noinput_{uuid.uuid4().hex}"

        # Human readable portion
        hr_label = ""
        # Create a deterministic id using hashing
        hash_id = self._init_hash()
        hash_id.update(f"{index}".encode())
        if prompt_wav is not None:
            hash_id.update(prompt_wav.numpy().data)
            hr_label += "_prompted"
        else:
            hr_label += "_unprompted"
        if conditions:
            encoded_json = json.dumps(conditions, sort_keys=True).encode()
            hash_id.update(encoded_json)
            cond_str = "-".join([f"{key}={slugify(value)}" for key, value in sorted(conditions.items())])
            cond_str = cond_str[:100]  # some raw text might be too long to be a valid filename
            cond_str = cond_str if len(cond_str) > 0 else "unconditioned"
            hr_label += f"_{cond_str}"
        else:
            hr_label += "_unconditioned"

        return hash_id.hexdigest() + hr_label

    def _store_audio(self, wav: torch.Tensor, stem_path: Path, overwrite: bool = False) -> Path:
        """Stores the audio with the given stem path using the XP's configuration.

        Args:
            wav (torch.Tensor): Audio to store.
            stem_path (Path): Path in sample output directory with file stem to use.
            overwrite (bool): When False (default), skips storing an existing audio file.
        Returns:
            Path: The path at which the audio is stored.
        """
        existing_paths = [path for path in stem_path.parent.glob(stem_path.stem + ".*") if path.suffix != ".json"]
        exists = len(existing_paths) > 0
        if exists and overwrite:
            logger.warning(f"Overwriting existing audio file with stem path {stem_path}")
        elif exists:
            return existing_paths[0]

        audio_path = audio_write(stem_path, wav, **self.xp.cfg.generate.audio)
        return audio_path

    def add_sample(
        self,
        sample_wav: torch.Tensor,
        epoch: int,
        index: int = 0,
        conditions: tp.Optional[tp.Dict[str, str]] = None,
        prompt_wav: tp.Optional[torch.Tensor] = None,
        ground_truth_wav: tp.Optional[torch.Tensor] = None,
        generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> Sample:
        """Adds a single sample.
        The sample is stored in the XP's sample output directory, under a corresponding epoch folder.
        Each sample is assigned an id which is computed using the input data. In addition to the
        sample itself, a json file containing associated metadata is stored next to it.

        Args:
            sample_wav (torch.Tensor): sample audio to store. Tensor of shape [channels, shape].
            epoch (int): current training epoch.
            index (int): helpful to differentiate samples from the same batch.
            conditions (dict[str, str], optional): conditioning used during generation.
            prompt_wav (torch.Tensor, optional): prompt used during generation. Tensor of shape [channels, shape].
            ground_truth_wav (torch.Tensor, optional): reference audio where prompt was extracted from.
                Tensor of shape [channels, shape].
            generation_args (dict[str, any], optional): dictionary of other arguments used during generation.
        Returns:
            Sample: The saved sample.
        """
        sample_id = self._get_sample_id(index, prompt_wav, conditions)
        reuse_id = self.map_reference_to_sample_id
        prompt, ground_truth = None, None
        if prompt_wav is not None:
            prompt_id = sample_id if reuse_id else self._get_tensor_id(prompt_wav.sum(0, keepdim=True))
            prompt_duration = prompt_wav.shape[-1] / self.xp.cfg.sample_rate
            prompt_path = self._store_audio(prompt_wav, self.base_folder / str(epoch) / "prompt" / prompt_id)
            prompt = ReferenceSample(prompt_id, str(prompt_path), prompt_duration)
        if ground_truth_wav is not None:
            ground_truth_id = sample_id if reuse_id else self._get_tensor_id(ground_truth_wav.sum(0, keepdim=True))
            ground_truth_duration = ground_truth_wav.shape[-1] / self.xp.cfg.sample_rate
            ground_truth_path = self._store_audio(ground_truth_wav, self.base_folder / "reference" / ground_truth_id)
            ground_truth = ReferenceSample(ground_truth_id, str(ground_truth_path), ground_truth_duration)
        sample_path = self._store_audio(sample_wav, self.base_folder / str(epoch) / sample_id, overwrite=True)
        duration = sample_wav.shape[-1] / self.xp.cfg.sample_rate
        sample = Sample(
            sample_id,
            str(sample_path),
            epoch,
            duration,
            conditions,
            prompt,
            ground_truth,
            generation_args,
        )
        self.samples.append(sample)
        with open(sample_path.with_suffix(".json"), "w") as f:
            json.dump(asdict(sample), f, indent=2)
        return sample

    def add_samples(
        self,
        samples_wavs: torch.Tensor,
        epoch: int,
        conditioning: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
        prompt_wavs: tp.Optional[torch.Tensor] = None,
        ground_truth_wavs: tp.Optional[torch.Tensor] = None,
        generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> tp.List[Sample]:
        """Adds a batch of samples.
        The samples are stored in the XP's sample output directory, under a corresponding
        epoch folder. Each sample is assigned an id which is computed using the input data and their batch index.
        In addition to the sample itself, a json file containing associated metadata is stored next to it.

        Args:
            sample_wavs (torch.Tensor): Batch of audio wavs to store. Tensor of shape [batch_size, channels, shape].
            epoch (int): Current training epoch.
            conditioning (list of dict[str, str], optional): List of conditions used during generation,
                one per sample in the batch.
            prompt_wavs (torch.Tensor, optional): Prompts used during generation. Tensor of shape
                [batch_size, channels, shape].
            ground_truth_wav (torch.Tensor, optional): Reference audio where prompts were extracted from.
                Tensor of shape [batch_size, channels, shape].
            generation_args (dict[str, Any], optional): Dictionary of other arguments used during generation.
        Returns:
            samples (list of Sample): The saved audio samples with prompts, ground truth and metadata.
        """
        samples = []
        for idx, wav in enumerate(samples_wavs):
            prompt_wav = prompt_wavs[idx] if prompt_wavs is not None else None
            gt_wav = ground_truth_wavs[idx] if ground_truth_wavs is not None else None
            conditions = conditioning[idx] if conditioning is not None else None
            samples.append(self.add_sample(wav, epoch, idx, conditions, prompt_wav, gt_wav, generation_args))
        return samples

    def get_samples(
        self,
        epoch: int = -1,
        max_epoch: int = -1,
        exclude_prompted: bool = False,
        exclude_unprompted: bool = False,
        exclude_conditioned: bool = False,
        exclude_unconditioned: bool = False,
    ) -> tp.Set[Sample]:
        """Returns a set of samples for this XP. Optionally, you can filter which samples to obtain.
        Please note that existing samples are loaded during the manager's initialization, and added samples through this
        manager are also tracked. Any other external changes are not tracked automatically, so creating a new manager
        is the only way detect them.

        Args:
            epoch (int): If provided, only return samples corresponding to this epoch.
            max_epoch (int): If provided, only return samples corresponding to the latest epoch that is <= max_epoch.
            exclude_prompted (bool): If True, does not include samples that used a prompt.
            exclude_unprompted (bool): If True, does not include samples that did not use a prompt.
            exclude_conditioned (bool): If True, excludes samples that used conditioning.
            exclude_unconditioned (bool): If True, excludes samples that did not use conditioning.
        Returns:
            Samples (set of Sample): The retrieved samples matching the provided filters.
        """
        if max_epoch >= 0:
            samples_epoch = max(sample.epoch for sample in self.samples if sample.epoch <= max_epoch)
        else:
            samples_epoch = self.latest_epoch if epoch < 0 else epoch
        samples = {
            sample
            for sample in self.samples
            if (
                (sample.epoch == samples_epoch)
                and (not exclude_prompted or sample.prompt is None)
                and (not exclude_unprompted or sample.prompt is not None)
                and (not exclude_conditioned or not sample.conditioning)
                and (not exclude_unconditioned or sample.conditioning)
            )
        }
        return samples


class SampleManagerfMRI(SampleManager):
    def __init__(self, xp: dora.XP, map_reference_to_sample_id: bool = False):
        super().__init__(xp, map_reference_to_sample_id)
        self.samples: tp.List[SamplefMRI] = []

        difumo_path = fetch_atlas_difumo(dimension=xp.cfg.space_dim, resolution_mm=2, legacy_format=False).maps
        self.maps_masker = NiftiMapsMasker(maps_img=nb.load(difumo_path))

        self.maps_masker.fit()

    def _store_fMRI(
        self,
        bold: torch.Tensor,
        tr: float,
        stem_path: Path,
        overwrite: bool = False,
        block_num: int = 4,
        gt_bold: tp.Optional[np.ndarray] = None,
        stats: tp.Optional[tp.Dict[str, torch.Tensor]] = None,
    ) -> tp.Tuple[Path, np.ndarray]:
        existing_paths = [path for path in stem_path.parent.glob(stem_path.stem + ".*") if path.suffix != ".json"]
        exists = len(existing_paths) > 0
        if exists and overwrite:
            logger.warning(f"Overwriting existing audio file with stem path {stem_path}")
        elif exists:
            return existing_paths[0], bold.cpu().numpy().T

        # Save fMRI data
        fMRI_path = Path(str(stem_path) + ".tar")
        fMRI_path.parent.mkdir(exist_ok=True, parents=True)

        if stats is not None:
            # denormalize
            bold = bold.cpu().numpy()
            bold = bold * stats["std"].cpu().numpy() + stats["mean"].cpu().numpy()
            bold = bold.T
        else:
            bold = bold.cpu().numpy().T

        sample_dict = {"__key__": fMRI_path.stem, "bold.pyd": bold, "t_r.pyd": tr}
        with wds.TarWriter(str(fMRI_path)) as sink:
            sink.write(sample_dict)

        # Create fMRI image
        if gt_bold is not None:
            fMRI_img_path = Path(str(stem_path) + ".png")
            s_time = time.time()

            time_step = bold.shape[0]
            time_slice = np.arange(0, time_step, (time_step - 1) / (block_num - 1)).astype(int)

            input_bold = bold[time_slice]
            gt_input_bold = gt_bold[time_slice]

            reconstructed_img = self.maps_masker.inverse_transform(input_bold)
            gt_reconstructed_img = self.maps_masker.inverse_transform(gt_input_bold)

            block_num *= 3

            fig, axes = plt.subplots(figsize=(block_num * 3, block_num * 2), ncols=2, nrows=block_num // 2)
            axes = axes.reshape(-1)
            for i, (step, cmap) in enumerate(zip(time_slice, iter_img(reconstructed_img))):
                plotting.plot_stat_map(
                    cmap,
                    figure=fig,
                    axes=axes[i],
                    colorbar=False,
                    title=f"Gen: Step {step+1}",
                )

            for i, (step, cmap) in enumerate(zip(time_slice, iter_img(gt_reconstructed_img))):
                plotting.plot_stat_map(
                    cmap,
                    figure=fig,
                    axes=axes[i + (block_num // 3)],
                    colorbar=False,
                    title=f"GT: Step {step+1}",
                )

            diff_from_gt_img = math_img("np.abs(img1 - img2)", img1=reconstructed_img, img2=gt_reconstructed_img)
            for i, (step, cmap) in enumerate(zip(time_slice, iter_img(diff_from_gt_img))):
                plotting.plot_stat_map(
                    cmap,
                    figure=fig,
                    axes=axes[i + 2 * (block_num // 3)],
                    colorbar=False,
                    title=f"Diff: Step {step+1}",
                )

            plt.savefig(fMRI_img_path)
            plt.close()

            logger.debug(f"Time to create fMRI image: {fMRI_path}:{time.time() - s_time}")
        return fMRI_path, bold

    def add_sample(
        self,
        sample_bold: torch.Tensor,
        tr: float,
        epoch: int,
        index: int = 0,
        conditions: tp.Optional[tp.Dict[str, str]] = None,
        prompt_bold: tp.Optional[torch.Tensor] = None,
        ground_truth_bold: tp.Optional[torch.Tensor] = None,
        generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
        stats: tp.Optional[tp.Dict[str, torch.Tensor]] = None,
    ) -> SamplefMRI:
        sample_id = self._get_sample_id(index, prompt_bold, conditions)
        reuse_id = self.map_reference_to_sample_id
        prompt, ground_truth = None, None
        if prompt_bold is not None:
            raise NotImplementedError("Prompting is not supported for fMRI samples.")
        gt_bold = None
        if ground_truth_bold is not None:
            ground_truth_id = sample_id if reuse_id else self._get_tensor_id(ground_truth_bold.sum(0, keepdim=True))
            ground_truth_duration = ground_truth_bold.shape[-1]
            ground_truth_path, gt_bold = self._store_fMRI(
                ground_truth_bold,
                tr,
                self.base_folder / "reference" / ground_truth_id,
                block_num=self.xp.cfg.generate.block_num,
                stats=stats,
            )
            ground_truth = ReferencefMRISample(ground_truth_id, str(ground_truth_path), ground_truth_duration, tr)
        sample_path, _ = self._store_fMRI(
            sample_bold,
            tr,
            self.base_folder / str(epoch) / sample_id,
            overwrite=True,
            gt_bold=gt_bold,
            block_num=self.xp.cfg.generate.block_num,
            stats=stats,
        )
        duration = sample_bold.shape[-1]
        sample = SamplefMRI(
            sample_id,
            str(sample_path),
            epoch,
            duration,
            tr,
            conditions,
            prompt,
            ground_truth,
            generation_args,
        )
        self.samples.append(sample)
        with open(sample_path.with_suffix(".json"), "w") as f:
            json.dump(asdict(sample), f, indent=2)
        return sample

    def add_samples(
        self,
        samples_bolds: torch.Tensor,
        tr: tp.List[float],
        epoch: int,
        conditioning: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
        prompt_bolds: tp.Optional[torch.Tensor] = None,
        ground_truth_bolds: tp.Optional[torch.Tensor] = None,
        generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
        stats: tp.Optional[tp.Dict[str, torch.Tensor]] = None,
    ) -> tp.List[Sample]:
        samples = []
        for idx, bold in enumerate(samples_bolds):
            prompt_bold = prompt_bolds[idx] if prompt_bolds is not None else None
            gt_bold = ground_truth_bolds[idx] if ground_truth_bolds is not None else None
            conditions = conditioning[idx] if conditioning is not None else None
            samples.append(
                self.add_sample(
                    bold,
                    tr[idx],
                    epoch,
                    idx,
                    conditions,
                    prompt_bold,
                    gt_bold,
                    generation_args,
                    stats,
                )
            )
        return samples


def slugify(value: tp.Any, allow_unicode: bool = False):
    """Process string for safer file naming.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def _match_stable_samples(
    samples_per_xp: tp.List[tp.Set[Sample]],
) -> tp.Dict[str, tp.List[Sample]]:
    # Create a dictionary of stable id -> sample per XP
    stable_samples_per_xp = [
        {sample.id: sample for sample in samples if sample.prompt is not None or sample.conditioning}
        for samples in samples_per_xp
    ]
    # Set of all stable ids
    stable_ids = {id for samples in stable_samples_per_xp for id in samples.keys()}
    # Dictionary of stable id -> list of samples. If an XP does not have it, assign None
    stable_samples = {id: [xp.get(id) for xp in stable_samples_per_xp] for id in stable_ids}
    # Filter out ids that contain None values (we only want matched samples after all)
    # cast is necessary to avoid mypy linter errors.
    return {id: tp.cast(tp.List[Sample], samples) for id, samples in stable_samples.items() if None not in samples}


def _match_unstable_samples(
    samples_per_xp: tp.List[tp.Set[Sample]],
) -> tp.Dict[str, tp.List[Sample]]:
    # For unstable ids, we use a sorted list since we'll match them in order
    unstable_samples_per_xp = [
        [sample for sample in sorted(samples, key=lambda x: x.id) if sample.prompt is None and not sample.conditioning]
        for samples in samples_per_xp
    ]
    # Trim samples per xp so all samples can have a match
    min_len = min([len(samples) for samples in unstable_samples_per_xp])
    unstable_samples_per_xp = [samples[:min_len] for samples in unstable_samples_per_xp]
    # Dictionary of index -> list of matched samples
    return {f"noinput_{i}": [samples[i] for samples in unstable_samples_per_xp] for i in range(min_len)}


def get_samples_for_xps(xps: tp.List[dora.XP], **kwargs) -> tp.Dict[str, tp.List[Sample]]:
    """Gets a dictionary of matched samples across the given XPs.
    Each dictionary entry maps a sample id to a list of samples for that id. The number of samples per id
    will always match the number of XPs provided and will correspond to each XP in the same order given.
    In other words, only samples that can be match across all provided XPs will be returned
    in order to satisfy this rule.

    There are two types of ids that can be returned: stable and unstable.
    * Stable IDs are deterministic ids that were computed by the SampleManager given a sample's inputs
      (prompts/conditioning). This is why we can match them across XPs.
    * Unstable IDs are of the form "noinput_{idx}" and are generated on-the-fly, in order to map samples
      that used non-deterministic, random ids. This is the case for samples that did not use prompts or
      conditioning for their generation. This function will sort these samples by their id and match them
      by their index.

    Args:
        xps: a list of XPs to match samples from.
        start_epoch (int): If provided, only return samples corresponding to this epoch or newer.
        end_epoch (int): If provided, only return samples corresponding to this epoch or older.
        exclude_prompted (bool): If True, does not include samples that used a prompt.
        exclude_unprompted (bool): If True, does not include samples that did not use a prompt.
        exclude_conditioned (bool): If True, excludes samples that used conditioning.
        exclude_unconditioned (bool): If True, excludes samples that did not use conditioning.
    """
    managers = [SampleManager(xp) for xp in xps]
    samples_per_xp = [manager.get_samples(**kwargs) for manager in managers]
    stable_samples = _match_stable_samples(samples_per_xp)
    unstable_samples = _match_unstable_samples(samples_per_xp)
    return dict(stable_samples, **unstable_samples)
