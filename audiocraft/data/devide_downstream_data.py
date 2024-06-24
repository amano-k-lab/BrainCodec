import argparse
import os
import random
import sys

import webdataset as wds
from torch import manual_seed

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, "../"))
from pathlib import Path

import tools
from tqdm import tqdm


def load_tarfiles(output_base, tarfile_paths, label_name):
    output_base.mkdir(parents=True, exist_ok=True)
    dataset = wds.WebDataset(tarfile_paths).decode("pil")
    for sample in tqdm(dataset):
        file_name = sample["__key__"]
        output = {
            "__key__": file_name,
            "bold.pyd": sample["bold.pyd"],
            "t_r.pyd": sample["t_r.pyd"],
            "label.pyd": sample[label_name],
        }
        with wds.TarWriter(str(output_base / (file_name + ".tar"))) as sink:
            sink.write(output)


def main():
    config = get_config()

    random.seed(config["seed"])
    manual_seed(config["seed"])

    tarfile_paths = tools.grab_tarfile_paths(config["data"])
    tarfile_paths_split = tools.split_tarfile_paths_train_val(
        tarfile_paths=tarfile_paths,
        n_train_subjects_per_dataset=config["n_train_subjects_per_dataset"],
        n_val_subjects_per_dataset=config["n_val_subjects_per_dataset"],
        n_test_subjects_per_dataset=config["n_test_subjects_per_dataset"],
        seed=config["seed"],
    )
    train_tarfile_paths = tarfile_paths_split["train"]
    validation_tarfile_paths = tarfile_paths_split["validation"]
    test_tarfile_paths = tarfile_paths_split["test"]
    total_num = len(train_tarfile_paths) + len(validation_tarfile_paths) + len(test_tarfile_paths)

    print(
        "train tarfile paths:",
        len(train_tarfile_paths),
        "(",
        len(train_tarfile_paths) / total_num,
        ")",
    )
    print(
        "valid tarfile paths:",
        len(validation_tarfile_paths),
        "(",
        len(validation_tarfile_paths) / total_num,
        ")",
    )
    print(
        "test  tarfile paths:",
        len(test_tarfile_paths),
        "(",
        len(test_tarfile_paths) / total_num,
        ")",
    )
    target_base_path = Path(config["output"])
    load_tarfiles(target_base_path / "train", train_tarfile_paths, config["label_name"])
    load_tarfiles(target_base_path / "valid", validation_tarfile_paths, config["label_name"])
    load_tarfiles(target_base_path / "test", test_tarfile_paths, config["label_name"])


def get_config():
    parser = argparse.ArgumentParser(description="run model training")

    parser.add_argument(
        "--seed",
        metavar="INT",
        default=1234,
        type=int,
        help="random seed (default: 1234)",
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        default="data/downstream/HCP",
        type=str,
        help="path to training data directory " "(default: data/upstream)",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="data/downstream_split/HCP",
        type=str,
        help="path to training data directory " "(default: data/upstream)",
    )
    parser.add_argument(
        "--label_name",
        metavar="STR",
        default="label_across_tasks.pyd",
        type=str,
    )
    parser.add_argument(
        "--n-train-subjects-per-dataset",
        metavar="INT",
        default=None,
        type=int,
        help="number of subjects per dataset that are "
        "randomly selected as train data. "
        "! overrides --frac-train-per-dataset and "
        "requires setting --n-train-subjects-per-dataset",
    )
    parser.add_argument(
        "--n-val-subjects-per-dataset",
        metavar="INT",
        default=10,
        type=int,
        help="number of subjects per dataset that are "
        "randomly selected as validation data. "
        "! overrides --frac-val-per-dataset and "
        "requires setting --n-train-subjects-per-dataset",
    )
    parser.add_argument(
        "--n-test-subjects-per-dataset",
        metavar="INT",
        default=20,
        type=int,
        help="number of subjects per dataset that are "
        "randomly selected as test data. "
        "! Test set is only created if this is set != -1",
    )
    parser.add_argument(
        "--sample-random-seq",
        metavar="BOOL",
        choices=("True", "False"),
        default="True",
        help="whether or not to randomly sample input sequences "
        "from BOLD --data during training "
        "(default: True). "
        "Range for randomly sampled sequence lengths specified by "
        "--seq-min and --seq-max",
    )
    parser.add_argument(
        "--seq-min",
        metavar="INT",
        default=10,
        type=int,
        help="minimum length of randomly sampled BOLD input sequences " "(in number of TRs; default: 10)",
    )
    parser.add_argument(
        "--seq-max",
        metavar="INT",
        default=50,
        type=int,
        help="maximum length of randomly sampled BOLD input sequences " "(in number of TRs; default: 50)",
    )
    config = vars(parser.parse_args())

    for arg in config:
        if config[arg] in ["True", "False"]:
            config[arg] = config[arg] == "True"

        elif config[arg] == "none":
            config[arg] = None

        elif "subjects_per_dataset" in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    return config


if __name__ == "__main__":
    main()
