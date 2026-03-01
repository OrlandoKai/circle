import argparse
import json
from pathlib import Path
from typing import Any
import copy

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-config",
        required=True,
        type=str,
        help="Base configuration file to clone",
    )

    parser.add_argument(
        "--new-config",
        required=True,
        type=str,
        help="New configuration file to create. If it starts with '>', it will replace the base config name entirely.",
    )

    parser.add_argument(
        "--new-task-name",
        required=False,
        type=str,
        default=None,
        help="New task name",
    )

    parser.add_argument(
        "--update-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to update existing configurations",
    )

    parser.add_argument(
        "--diff",
        type=str,
        default="config-diff.json",
        help="JSON file with the differences to apply to the base config",
    )

    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use this flag to only print the changes without writing them to file",
    )

    args = parser.parse_args()

    return args


def _deep_update(original: dict, updates: dict) -> dict:
    """Recursively merge updates into original, preserving keys not in updates."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            _deep_update(original[key], value)
        else:
            # Preserve quoted strings from updates
            if isinstance(value, str):
                original[key] = DoubleQuotedScalarString(value)
            else:
                original[key] = value
    return original


def _bool_representer(dumper, data: bool) -> Any:
    """Represent boolean values as 'True'/'False'."""
    if data is True:
        return dumper.represent_scalar("tag:yaml.org,2002:bool", "True")
    if data is False:
        return dumper.represent_scalar("tag:yaml.org,2002:bool", "False")


def _yaml_setup() -> YAML:
    """Setup YAML parser with desired options."""
    yaml = YAML()

    # Disable line wrapping
    yaml.width = float("inf")

    # Keep original string quotes
    yaml.preserve_quotes = True

    # Register the custom boolean representer
    yaml.representer.add_representer(bool, _bool_representer)

    return yaml


def _make_int_keys(diff: dict) -> dict:
    """Convert string keys that are integers to actual integers."""
    new_diff = {}
    for k, v in diff.items():
        if isinstance(v, dict):
            v = _make_int_keys(v)
        try:
            int_k = int(k)
            new_diff[int_k] = v
        except ValueError:
            new_diff[k] = v
    return new_diff


def main(args: argparse.Namespace) -> None:
    yaml = _yaml_setup()

    # Load the diff file
    with open(args.diff) as f:
        edit = json.load(f)
    edit = _make_int_keys(edit)

    # Get the folder containing all the tasks
    tasks_parents = Path("src") / "data" / "tasks" / "_classification"

    # For each parent task (i.e., the datasets/benchmarks)
    for task_parent in tasks_parents.iterdir():
        # Construct the path to the base task
        print(f"Task: {task_parent.name}")
        base_task = task_parent / (args.base_config + ".yaml")

        if not base_task.exists():
            print(f"\t> Skipping {base_task}, does not exist")
            continue

        # Construct the path to the new task
        if args.new_config.startswith(">"):
            new_task = task_parent / (args.new_config[1:] + ".yaml")
        else:
            new_task = task_parent / (args.base_config + args.new_config + ".yaml")

        # If the new task already exists and we don't want to update existing tasks, skip it
        if new_task.exists() and not args.update_existing:
            print(f"\t> Skipping {new_task}, already exists and update_existing is False")
            continue

        # Load the base task
        with open(base_task) as f:
            base_task_dict = yaml.load(f)

        # Edit base task with `edit` info
        edit_custom = copy.deepcopy(edit)
        # Find any strings that is "$TASK_NAME" and replace it with `task_parent.name`
        def replace_task_name(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    replace_task_name(v)
                elif isinstance(v, str) and "$TASK_NAME" in v:
                    d[k] = v.replace("$TASK_NAME", task_parent.name)
        replace_task_name(edit_custom)

        _deep_update(base_task_dict, edit_custom)
        if args.new_task_name is not None:
            base_task_dict["task"] = task_parent.name + "_" + args.new_task_name
        else:
            base_task_dict["task"] = base_task_dict["task"] + args.new_config

        if args.dry_run:
            if new_task.exists():
                print(f"\t> Would update {new_task}")
            else:
                print(f"\t> Would create {new_task}")
            continue

        # Save the new file while keeping the same order and spacing as the original file
        with open(new_task, "w") as f:
            yaml.dump(base_task_dict, f)


if __name__ == "__main__":
    args = _get_args()
    main(args)
