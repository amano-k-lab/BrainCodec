import argparse
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, "../"))
from pathlib import Path

import tools


def main():
    config = get_config()
    tarfile_paths = tools.grab_tarfile_paths(config["data"])
    tarfile_paths_split = tools.split_tarfile_paths_train_val(
        tarfile_paths=tarfile_paths,
        frac_val_per_dataset=config["frac_val_per_dataset"],
        frac_test_per_dataset=config["frac_test_per_dataset"],
        n_val_subjects_per_dataset=config["n_val_subjects_per_dataset"],
        n_test_subjects_per_dataset=config["n_test_subjects_per_dataset"],
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
    print("Create symlink to tarfiles in output directory")
    for phase in ["train", "valid", "test"]:
        target_base_path = Path(config["output"]) / phase
        target_base_path.mkdir(parents=True, exist_ok=True)
        if phase == "train":
            for source_path in train_tarfile_paths:
                source_path = Path(source_path)
                target_path = target_base_path / source_path.name
                target_path.symlink_to(source_path.resolve())
        elif phase == "valid":
            for source_path in validation_tarfile_paths:
                source_path = Path(source_path)
                target_path = target_base_path / source_path.name
                target_path.symlink_to(source_path.resolve())
        else:
            for source_path in test_tarfile_paths:
                source_path = Path(source_path)
                target_path = target_base_path / source_path.name
                target_path.symlink_to(source_path.resolve())


def get_config():
    parser = argparse.ArgumentParser(description="run model training")

    parser.add_argument(
        "--data",
        metavar="DIR",
        default="data/upstream",
        type=str,
        help="path to training data directory " "(default: data/upstream)",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="data/upstream_split",
        type=str,
        help="path to training data directory " "(default: data/upstream)",
    )
    parser.add_argument(
        "--frac-val-per-dataset",
        metavar="FLOAT",
        default=0.04,
        type=float,
        help="fraction of fMRI runs per dataset that " "are randomly selected as validation data " "(default: 0.05)",
    )
    parser.add_argument(
        "--frac-test-per-dataset",
        metavar="FLOAT",
        default=0.01,
        type=float,
        help="fraction of fMRI runs per dataset that " "are randomly selected as validation data " "(default: 0.05)",
    )
    parser.add_argument(
        "--n-val-subjects-per-dataset",
        metavar="INT",
        default=-1,
        type=int,
        help="number of subjects per dataset that are "
        "randomly selected as validation data. "
        "! overrides --frac-val-per-dataset and "
        "requires setting --n-train-subjects-per-dataset",
    )
    parser.add_argument(
        "--n-test-subjects-per-dataset",
        metavar="INT",
        default=-1,
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
