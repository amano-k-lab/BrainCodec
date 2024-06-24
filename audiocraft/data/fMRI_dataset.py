import argparse
import copy
import gzip
import io
import json
import logging
import os
import random
import sys
import tarfile
import typing as tp
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..modules.conditioners import ConditioningAttributes
from .audio_dataset import (AudioDataset, BaseInfo, _resolve_audio_meta,
                            save_audio_meta)
from .zip import PathInZip

DEFAULT_EXTS = [".tar"]


logger = logging.getLogger(__name__)


@dataclass(order=True)
class fMRIMeta(BaseInfo):
    path: str
    tr: tp.Optional[float] = None
    duration: tp.Optional[int] = None
    weight: tp.Optional[float] = None  # Used for _get_sampling_probabilities
    label: tp.Optional[int] = None  # For downstream task
    # info_path is used to load additional information about the audio file that is stored in zip files.
    info_path: tp.Optional[PathInZip] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        if "info_path" in base and base["info_path"] is not None:
            base["info_path"] = PathInZip(base["info_path"])
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d["info_path"] is not None:
            d["info_path"] = str(d["info_path"])
        return d


@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: fMRIMeta
    seek_time: float
    duration: int  # actual number of duration without padding
    total_duration: int  # total number of duration, padding included
    tr: float  # repetition time
    tr_precision: float  # repetition time precision
    max_tr: float  # max repetition time
    space_dim: int  # number of voxels

    def to_condition_attributes(self) -> ConditioningAttributes:
        return ConditioningAttributes()


def _preprocess_sample(sample: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    t_r = float(sample["t_r.pyd"])
    bold = np.array(sample["bold.pyd"]).astype(float).T
    if "label.pyd" in sample.keys():
        label = int(sample["label.pyd"])
    else:
        label = None
    return dict(bold=bold, tr=t_r, label=label)


def load_data_from_tar(tar_file_path: str) -> tp.Dict[str, np.ndarray]:
    data = {}
    with tarfile.open(tar_file_path, "r") as tar:
        for member in tar.getmembers():
            # ファイルを読み込む
            file_data = tar.extractfile(member)
            file_name = member.name.split(".", 1)[-1]
            if file_name in ["bold.pyd", "t_r.pyd", "label.pyd"]:
                # Numpy配列としてデータを読み込む
                np_array = np.load(io.BytesIO(file_data.read()), allow_pickle=True)
                data[file_name] = np_array
    return _preprocess_sample(data)


def fMRI_read(
    file_path: str,
    seek_time: int = 0,
    mean: tp.Optional[torch.Tensor] = None,
    std: tp.Optional[torch.Tensor] = None,
) -> tp.Tuple[torch.Tensor, float]:
    output = load_data_from_tar(file_path)
    bold = torch.from_numpy(output["bold"]).float()

    if mean is not None:
        bold = (bold - mean) / std

    return (
        bold[:, seek_time:],
        output["tr"],
        output["label"],
    )  # bold data should be [space_dim, time_dim]


def _get_fMRI_meta(file_path: str, minimal: bool = True) -> fMRIMeta:
    """fMRIMeta from a path to an audio file.

    Args:
        file_path (str): Resolved path of valid audio file.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
    Returns:
        fMRIMeta: Audio file path and its metadata.
    """
    tr: tp.Optional[float] = None
    duration: tp.Optional[int] = None
    if not minimal:
        bold, tr, label = fMRI_read(file_path)
        tr = float(tr)
        duration = int(bold.shape[-1])
    return fMRIMeta(file_path, tr, duration, label=label)


def find_fMRI_files(
    path: tp.Union[Path, str],
    exts: tp.List[str] = DEFAULT_EXTS,
    resolve: bool = True,
    minimal: bool = True,
    progress: bool = False,
    workers: int = 0,
) -> tp.List[fMRIMeta]:
    """Build a list of fMRIMeta from a given path,
    collecting relevant audio files and fetching meta info.

    Args:
        path (str or Path): Path to folder containing audio files.
        exts (list of str): List of file extensions to consider for audio files.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
        progress (bool): Whether to log progress on audio files collection.
        workers (int): number of parallel workers, if 0, use only the current thread.
    Returns:
        list of fMRIMeta: List of audio file path and its metadata.
    """
    fMRI_files = []
    futures: tp.List[Future] = []
    pool: tp.Optional[ThreadPoolExecutor] = None
    with ExitStack() as stack:
        if workers > 0:
            pool = ThreadPoolExecutor(workers)
            stack.enter_context(pool)

        if progress:
            print("Finding fMRI files...")
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = Path(root) / file
                if full_path.suffix.lower() in exts:
                    fMRI_files.append(full_path)
                    if pool is not None:
                        futures.append(pool.submit(_get_fMRI_meta, str(fMRI_files[-1]), minimal))
                    if progress:
                        print(format(len(fMRI_files), " 8d"), end="\r", file=sys.stderr)

        if progress:
            print("Getting fMRI metadata...")
        meta: tp.List[fMRIMeta] = []
        for idx, file_path in enumerate(fMRI_files):
            try:
                if pool is None:
                    m = _get_fMRI_meta(str(file_path), minimal)
                else:
                    m = futures[idx].result()
                if resolve:
                    m = _resolve_audio_meta(m)
            except Exception as err:
                print("Error with", str(file_path), err, file=sys.stderr)
                continue
            meta.append(m)
            if progress:
                print(
                    format((1 + idx) / len(fMRI_files), " 3.1%"),
                    end="\r",
                    file=sys.stderr,
                )
    meta.sort()
    return meta


class fMRIDataset(AudioDataset):
    """Base fMRI dataset.
    Args:
        meta (list of AudioMeta): List of audio files metadata.
        segment_duration (float, optional): Optional segment duration of audio to load.
            If not specified, the dataset will load the full audio segment from the file.
        shuffle (bool): Set to `True` to have the data reshuffled at every epoch.
        sample_rate (int): Target sample rate of the loaded audio samples.
        channels (int): Target number of channels of the loaded audio samples.
        sample_on_duration (bool): Set to `True` to sample segments with probability
            dependent on audio file duration. This is only used if `segment_duration` is provided.
        sample_on_weight (bool): Set to `True` to sample segments using the `weight` entry of
            `AudioMeta`. If `sample_on_duration` is also True, the actual weight will be the product
            of the file duration and file weight. This is only used if `segment_duration` is provided.
        min_segment_ratio (float): Minimum segment ratio to use when the audio file
            is shorter than the desired segment.
        max_read_retry (int): Maximum number of retries to sample an audio segment from the dataset.
        return_info (bool): Whether to return the wav only or return wav along with segment info and metadata.
        min_audio_duration (float, optional): Minimum audio file duration, in seconds, if provided
            audio shorter than this will be filtered out.
        max_audio_duration (float, optional): Maximal audio file duration in seconds, if provided
            audio longer than this will be filtered out.
        shuffle_seed (int): can be used to further randomize
        load_wav (bool): if False, skip loading the wav but returns a tensor of 0
            with the expected segment_duration (which must be provided if load_wav is False).
        permutation_on_files (bool): only if `sample_on_weight` and `sample_on_duration`
            are False. Will ensure a permutation on files when going through the dataset.
            In that case the epoch number must be provided in order for the model
            to continue the permutation across epochs. In that case, it is assumed
            that `num_samples = total_batch_size * num_updates_per_epoch`, with
            `total_batch_size` the overall batch size accounting for all gpus.
    """

    def __init__(
        self,
        meta: tp.List[fMRIMeta],
        segment_duration: tp.Optional[float] = None,
        shuffle: bool = True,
        num_samples: int = 10_000,
        space_dim: int = 1024,
        pad: bool = True,
        sample_on_duration: bool = True,
        sample_on_weight: bool = True,
        min_segment_ratio: float = 0.5,
        max_read_retry: int = 10,
        return_info: bool = False,
        min_fMRI_duration: tp.Optional[float] = None,
        max_fMRI_duration: tp.Optional[float] = None,
        tr_precision: float = 0.2,
        max_tr: float = 300.0,
        shuffle_seed: int = 0,
        load_fMRI: bool = True,
        permutation_on_files: bool = False,
        normalize: bool = False,
        split: str = "train",
        root_path: Path = None,
        sample_w_previous_research: bool = True,
    ):
        assert len(meta) > 0, "No fMRI meta provided to fMRIDataset. Please check loading of audio meta."
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.max_fMRI_duration = max_fMRI_duration
        self.min_fMRI_duration = min_fMRI_duration
        self.tr_precision = tr_precision
        self.max_tr = max_tr
        if self.min_fMRI_duration is not None and self.max_fMRI_duration is not None:
            assert self.min_fMRI_duration <= self.max_fMRI_duration
        self.meta: tp.List[fMRIMeta] = self._filter_tr(self._filter_duration(meta))
        assert len(self.meta)  # Fail fast if all data has been filtered.
        self.total_duration = sum(d.duration for d in self.meta)

        if segment_duration is None:
            num_samples = len(self.meta)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.space_dim = space_dim
        self.pad = pad
        self.sample_on_weight = sample_on_weight
        self.sample_on_duration = sample_on_duration
        self.sampling_probabilities = self._get_sampling_probabilities()
        self.max_read_retry = max_read_retry
        self.return_info = True  # always return info
        self.shuffle_seed = shuffle_seed
        self.current_epoch: tp.Optional[int] = None
        self.load_fMRI = load_fMRI
        self.sample_w_previous_research = sample_w_previous_research
        if not load_fMRI:
            assert segment_duration is not None
        self.permutation_on_files = permutation_on_files
        if permutation_on_files:
            assert not self.sample_on_duration
            assert not self.sample_on_weight
            assert self.shuffle

        self.normalize_mean = self.normalize_std = None
        if normalize is True:
            if root_path is None:
                raise ValueError("root_path should not be None.")
            normalize_path = root_path.parent / "train" / "normalize.jsonl.gz"
            open_fn = gzip.open if str(normalize_path).lower().endswith(".gz") else open
            if normalize_path.exists():
                logger.info("Loading normalization parameters")
                with open_fn(normalize_path, "rb") as fp:  # type: ignore
                    lines = fp.readlines()
                d = json.loads(lines[0])
                self.normalize_mean = torch.tensor(d["mean"])
                self.normalize_std = torch.tensor(d["std"])
            else:
                if split == "train":
                    logger.info("Normalizing fMRI data")
                    all_fMRI_data = torch.cat(
                        [fMRI_read(file_meta.path)[0] for file_meta in self.meta],
                        dim=-1,
                    )
                    self.normalize_mean = torch.mean(all_fMRI_data, dim=-1, keepdim=True)
                    self.normalize_std = torch.std(all_fMRI_data, dim=-1, keepdim=True)
                    save_stats = {
                        "mean": self.normalize_mean.numpy().tolist(),
                        "std": self.normalize_std.numpy().tolist(),
                    }
                    with open_fn(normalize_path, "wb") as fp:  # type: ignore
                        json_str = json.dumps(save_stats)
                        json_bytes = json_str.encode("utf-8")
                        fp.write(json_bytes)
                    logger.info("Finish normalizing fMRI data")
                else:
                    raise ValueError("The other split is called before train split")

    def _fMRI_read(self, path: str):
        # Override this method in subclass if needed.
        if self.load_fMRI:
            return fMRI_read(path)
        else:
            assert self.segment_duration is not None
            return torch.zeros(self.segment_duration, self.space_dim), self.tr_precision

    def __getitem__(self, index: int) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, SegmentInfo]]:
        if self.segment_duration is None:
            file_meta = self.meta[index]
            out, tr, _ = fMRI_read(file_meta.path, mean=self.normalize_mean, std=self.normalize_std)
            segment_info = SegmentInfo(
                file_meta,
                seek_time=0.0,
                duration=int(out.shape[-1]),
                total_duration=int(out.shape[-1]),
                tr=tr,
                tr_precision=self.tr_precision,
                max_tr=self.max_tr,
                space_dim=out.shape[0],
            )
        else:
            rng = torch.Generator()
            if self.shuffle:
                # We use index, plus extra randomness, either totally random if we don't know the epoch.
                # otherwise we make use of the epoch number and optional shuffle_seed.
                if self.current_epoch is None:
                    rng.manual_seed(index + self.num_samples * random.randint(0, 2**24))
                else:
                    rng.manual_seed(index + self.num_samples * (self.current_epoch + self.shuffle_seed))
            else:
                # We only use index
                rng.manual_seed(index)

            for retry in range(self.max_read_retry):
                file_meta = self.sample_file(index, rng)
                # We add some variance in the file position even if audio file is smaller than segment
                # without ending up with empty segments
                if self.sample_w_previous_research is True:
                    min_length = min(
                        int(self.segment_duration * self.min_segment_ratio),
                        file_meta.duration,
                    )
                    max_length = min(int(self.segment_duration), file_meta.duration)

                    if min_length < max_length:
                        c_length = int(torch.rand(1, generator=rng).item() * (max_length - min_length)) + min_length
                    else:
                        c_length = max_length

                    if c_length < file_meta.duration:
                        seek_time = int(torch.rand(1, generator=rng).item() * (file_meta.duration - c_length))
                    else:
                        seek_time = 0
                    try:
                        out, tr, _ = fMRI_read(
                            file_meta.path,
                            seek_time=seek_time,
                            mean=self.normalize_mean,
                            std=self.normalize_std,
                        )
                        out = out[:, :c_length]
                        duration = int(out.shape[-1])
                        target_duration = self.segment_duration
                        if self.pad:
                            out = F.pad(out, (0, target_duration - duration))
                        segment_info = SegmentInfo(
                            file_meta,
                            seek_time,
                            duration=duration,
                            total_duration=target_duration,
                            tr=tr,
                            tr_precision=self.tr_precision,
                            max_tr=self.max_tr,
                            space_dim=out.shape[0],
                        )
                    except Exception as exc:
                        logger.warning("Error opening file %s: %r", file_meta.path, exc)
                        if retry == self.max_read_retry - 1:
                            raise
                    else:
                        break
                else:
                    max_seek = max(
                        0,
                        file_meta.duration - self.segment_duration * self.min_segment_ratio,
                    )
                    seek_time = int(torch.rand(1, generator=rng).item() * max_seek)
                    try:
                        out, tr, _ = fMRI_read(
                            file_meta.path,
                            seek_time=seek_time,
                            mean=self.normalize_mean,
                            std=self.normalize_std,
                        )
                        duration = int(out.shape[-1])
                        target_duration = self.segment_duration
                        if self.pad:
                            out = F.pad(out, (0, target_duration - duration))
                        segment_info = SegmentInfo(
                            file_meta,
                            seek_time,
                            duration=duration,
                            total_duration=target_duration,
                            tr=tr,
                            tr_precision=self.tr_precision,
                            max_tr=self.max_tr,
                            space_dim=out.shape[0],
                        )
                    except Exception as exc:
                        logger.warning("Error opening file %s: %r", file_meta.path, exc)
                        if retry == self.max_read_retry - 1:
                            raise
                    else:
                        break

        if self.return_info:
            # Returns the wav and additional information on the wave segment
            return out, segment_info
        else:
            return out

    def collater(self, samples):
        """The collater function has to be provided to the dataloader
        if fMRIDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        if self.segment_duration is None and len(samples) > 1:
            assert self.pad, "Must allow padding when batching examples of different durations."

        # In this case the audio reaching the collater is of variable length as segment_duration=None.
        to_pad = self.segment_duration is None and self.pad
        if to_pad:
            max_len = max([len(out) for out, _ in samples])

            def _pad_out(out):
                return F.pad(out, (0, max_len - len(out)))

        if self.return_info:
            if len(samples) > 0:
                assert len(samples[0]) == 2
                assert isinstance(samples[0][0], torch.Tensor)
                assert isinstance(samples[0][1], SegmentInfo)

            outs = [out for out, _ in samples]
            segment_infos = [copy.deepcopy(info) for _, info in samples]

            if to_pad:
                # Each wav could be of a different duration as they are not segmented.
                for i in range(len(samples)):
                    # Determines the total length of the signal with padding, so we update here as we pad.
                    segment_infos[i].total_duration = max_len
                    outs[i] = _pad_out(outs[i])

            out = torch.stack(outs)
            return out, segment_infos
        else:
            assert isinstance(samples[0], torch.Tensor)
            if to_pad:
                samples = [_pad_out(s) for s in samples]
            return torch.stack(samples)

    def _filter_duration(self, meta: tp.List[fMRIMeta]) -> tp.List[fMRIMeta]:
        """Filters out audio files with audio durations that will not allow to sample examples from them."""
        orig_len = len(meta)

        # Filter data that is too short.
        if self.min_fMRI_duration is not None:
            meta = [m for m in meta if m.duration >= self.min_fMRI_duration]

        # Filter data that is too long.
        if self.max_fMRI_duration is not None:
            meta = [m for m in meta if m.duration <= self.max_fMRI_duration]

        filtered_len = len(meta)
        removed_percentage = 100 * (1 - float(filtered_len) / orig_len)
        msg = "Removed %.2f percent of the data because it was too short or too long." % removed_percentage
        if removed_percentage < 10:
            logging.debug(msg)
        else:
            logging.warning(msg)
        return meta

    def _filter_tr(self, meta: tp.List[fMRIMeta]) -> tp.List[fMRIMeta]:
        """Filters out audio files with audio durations that will not allow to sample examples from them."""
        orig_len = len(meta)

        # Filter data that is too short.
        if self.max_tr is not None:
            meta = [m for m in meta if (m.tr * self.segment_duration) <= self.max_tr]
        else:
            raise ValueError("max_tr must be provided")

        filtered_len = len(meta)
        removed_percentage = 100 * (1 - float(filtered_len) / orig_len)
        msg = "Removed %.2f percent of the data because it was too big tr." % removed_percentage
        if removed_percentage < 10:
            logging.debug(msg)
        else:
            logging.warning(msg)
        return meta

    @classmethod
    def from_meta(cls, root: tp.Union[str, Path], **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        kwargs["root_path"] = root
        if root.is_dir():
            if (root / "data.jsonl").exists():
                root = root / "data.jsonl"
            elif (root / "data.jsonl.gz").exists():
                root = root / "data.jsonl.gz"
            else:
                raise ValueError(
                    "Don't know where to read metadata from in the dir. "
                    "Expecting either a data.jsonl or data.jsonl.gz file but none found."
                )
        meta = load_fMRI_meta(root)
        return cls(meta, **kwargs)

    @classmethod
    def from_path(
        cls,
        root: tp.Union[str, Path],
        minimal_meta: bool = True,
        exts: tp.List[str] = DEFAULT_EXTS,
        **kwargs,
    ):
        """Instantiate AudioDataset from a path containing (possibly nested) audio files.

        Args:
            root (str or Path): Path to root folder containing audio files.
            minimal_meta (bool): Whether to only load minimal metadata or not.
            exts (list of str): Extensions for audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_file():
            meta = load_fMRI_meta(root, resolve=True)
        else:
            meta = find_fMRI_files(root, exts, minimal=minimal_meta, resolve=True)
        return cls(meta, **kwargs)


def load_fMRI_meta(path: tp.Union[str, Path], resolve: bool = True, fast: bool = True) -> tp.List[fMRIMeta]:
    """Load list of fMRIMeta from an optionally compressed json file.

    Args:
        path (str or Path): Path to JSON file.
        resolve (bool): Whether to resolve the path from AudioMeta (default=True).
        fast (bool): activates some tricks to make things faster.
    Returns:
        list of fMRIMeta: List of audio file path and its total duration.
    """
    open_fn = gzip.open if str(path).lower().endswith(".gz") else open
    with open_fn(path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = fMRIMeta.from_dict(d)
        if resolve:
            m = _resolve_audio_meta(m, fast=fast)
        meta.append(m)
    return meta


def main():
    """
    python -m audiocraft.data.fMRI_dataset ../learning-from-brains/data/downstream_split_hcp_n-48/HCP/valid egs/fMRI_downstream_tmp/HCP_n-48/valid/data.jsonl.gz --complete --resolve
    """
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser(prog="fMRI_dataset", description="Generate .jsonl files by scanning a folder.")
    parser.add_argument("root", help="Root folder with all the fMRI files")
    parser.add_argument("output_meta_file", help="Output file to store the metadata, ")
    parser.add_argument(
        "--complete",
        action="store_false",
        dest="minimal",
        default=True,
        help="Retrieve all metadata, even the one that are expansive " "to compute (e.g. normalization).",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        default=False,
        help="Resolve the paths to be absolute and with no symlinks.",
    )
    parser.add_argument("--workers", default=10, type=int, help="Number of workers.")
    parser.add_argument("--temp_for_weight", default=1, type=float)
    args = parser.parse_args()
    meta = find_fMRI_files(
        args.root,
        DEFAULT_EXTS,
        progress=True,
        resolve=args.resolve,
        minimal=args.minimal,
        workers=args.workers,
    )

    labels = []
    durations = []
    trs = []
    for m in meta:
        labels.append(m.label)
        durations.append(m.duration)
        trs.append(m.tr)
    print("label set: ", set(labels), " length: ", len(set(labels)))
    print("duration set: ", set(durations))
    print("tr set: ", set(trs))
    print("Update weight")
    number_of_labels = Counter(labels)
    print("label nums: ", number_of_labels)
    min_value = min(number_of_labels.values())
    number_of_labels = {k: min_value / v for k, v in number_of_labels.items()}
    print("label weight before softmax: ", number_of_labels)
    label_weight_values = np.array(list(number_of_labels.values())) / args.temp_for_weight
    after_softmax = np.exp(label_weight_values) / np.sum(np.exp(label_weight_values))
    for k, v in zip(number_of_labels.keys(), after_softmax):
        number_of_labels[k] = v
    print("weights: ", number_of_labels)
    for m in meta:
        m.weight = number_of_labels[m.label]
    save_audio_meta(args.output_meta_file, meta)


if __name__ == "__main__":
    main()
