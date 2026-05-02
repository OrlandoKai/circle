from functools import lru_cache
from pathlib import Path

import datasets
import pandas as pd
from PIL import Image

__all__ = [
    "ATTRIBUTES",
    "doc_to_text",
    "doc_to_text_with_context_top3",
    "doc_to_text_with_context_top5",
    "doc_to_text_with_context_top8",
    "doc_to_text_multi_round",
    "doc_to_visual",
    "doc_to_visual_with_context_top3",
    "doc_to_visual_with_context_top5",
    "doc_to_visual_with_context_top8",
    "doc_to_target",
    "doc_to_target_dummy",
    "process_docs_exclude_context_top3",
    "process_docs_exclude_context_top5",
    "process_docs_exclude_context_top8",
    "download",
]

ATTRIBUTES = [
    "TonguePale",
    "TipSideRed",
    "Spot",
    "Ecchymosis",
    "Crack",
    "Toothmark",
    "FurThick",
    "FurYellow",
]

_CONTEXT_TOP3_CSV = "train_fold1_min_top3_exact_AM.csv"
_CONTEXT_TOP5_CSV = "train_fold1_min_top5_exact_AM.csv"
_CONTEXT_TOP8_CSV = "train_fold1_min_top8_exact_AM.csv"

_DEFAULT_SINGLE_ROUND_PROMPT = (
    "You are an expert tongue diagnosis assistant.\n"
    "Use the reference examples to calibrate the decision boundary between 0 and 1 for each "
    "attribute in this dataset context.\n"
    "Then classify only the TARGET image.\n\n"
    "Output contract:\n"
    "- Output exactly 8 digits separated by single spaces.\n"
    "- Each digit must be 0 or 1.\n"
    "- Do not output any explanation or extra text.\n"
    "- Fixed order: TonguePale TipSideRed Spot Ecchymosis Crack Toothmark FurThick FurYellow"
)


def _image_root(task_root: Path) -> Path:
    root = task_root / "release" / "origin"
    if not root.exists():
        raise FileNotFoundError(f"Missing image root directory: {root}")
    return root


def _normalize_image_key(image_rel_path: str) -> str:
    # Dataset entries may contain legacy folder prefixes; runtime storage is flat.
    return Path(str(image_rel_path)).name


def _context_csv_path(task_root: Path, csv_name: str) -> Path:
    path = task_root / "release" / "list" / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Missing context CSV file: {path}")
    return path


@lru_cache(maxsize=8)
def _load_context_rows(task_root_str: str, csv_name: str) -> tuple[tuple[str, str], ...]:
    task_root = Path(task_root_str)
    csv_path = _context_csv_path(task_root, csv_name)

    required_cols = ["image_path", *ATTRIBUTES]
    df = pd.read_csv(csv_path, usecols=required_cols)
    rows: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        label = " ".join(str(int(row[attr])) for attr in ATTRIBUTES)
        rows.append((str(row["image_path"]), label))
    return tuple(rows)


@lru_cache(maxsize=8)
def _context_image_keys(task_root_str: str, csv_name: str) -> frozenset[str]:
    rows = _load_context_rows(task_root_str, csv_name)
    return frozenset(_normalize_image_key(image_rel_path) for image_rel_path, _ in rows)


def _doc_image_key(doc: dict) -> str:
    if "visual" in doc:
        return _normalize_image_key(str(doc["visual"]))
    if "image_path" in doc:
        return _normalize_image_key(str(doc["image_path"]))
    raise KeyError("Expected one of ['visual', 'image_path'] in document")


def _get_filtered_context_rows(task_root: Path, csv_name: str, target_key: str) -> list[tuple[str, str]]:
    rows = list(_load_context_rows(str(task_root), csv_name))
    return [(image_rel_path, label) for image_rel_path, label in rows if _normalize_image_key(image_rel_path) != target_key]


@lru_cache(maxsize=4)
def _build_image_index(task_root_str: str) -> dict[str, str]:
    task_root = Path(task_root_str)
    image_root = _image_root(task_root)
    index: dict[str, str] = {}

    for image_file in image_root.iterdir():
        if not image_file.is_file():
            continue

        key = image_file.name
        if key in index and index[key] != str(image_file):
            raise ValueError(f"Duplicate filename in flat image root: {key}")
        index[key] = str(image_file)

    if len(index) == 0:
        raise FileNotFoundError(f"No image files found under: {image_root}")

    return index


def _resolve_image_path(task_root: Path, image_rel_path: str) -> Path:
    index = _build_image_index(str(task_root))
    key = _normalize_image_key(image_rel_path)
    image_path = index.get(key)
    if image_path is None:
        raise FileNotFoundError(
            "Missing image file for "
            f"'{image_rel_path}' (normalized key: '{key}') in flat root "
            f"'{_image_root(task_root)}'"
        )
    return Path(image_path)


def doc_to_text(doc: dict, model_specific_kwargs: dict) -> str:
    """Build the text prompt for a TongeDx2 sample."""
    pre_prompt = model_specific_kwargs.get("pre_prompt", "")
    prompt = model_specific_kwargs.get("prompt", "")
    post_prompt = model_specific_kwargs.get("post_prompt", "")

    return pre_prompt + prompt + post_prompt


def _doc_to_text_with_context(doc: dict, model_specific_kwargs: dict, context_csv_name: str) -> str:
    task_root = Path(__file__).resolve().parents[1]
    target_key = _doc_image_key(doc)
    rows = _get_filtered_context_rows(task_root, context_csv_name, target_key)

    pre_prompt = model_specific_kwargs.get("pre_prompt", "")
    prompt = model_specific_kwargs.get("prompt", "") or _DEFAULT_SINGLE_ROUND_PROMPT
    post_prompt = model_specific_kwargs.get("post_prompt", "")

    if rows:
        label_lines = [f"Reference {idx + 1}: {label}" for idx, (_, label) in enumerate(rows)]
        context_block = (
            "Reference labels for the first {} image(s), in order:\n{}\n\n"
            "The last image is the TARGET image and has no label."
        ).format(len(rows), "\n".join(label_lines))
    else:
        context_block = "No reference samples remain after excluding TARGET overlap. Classify only the TARGET image."

    return pre_prompt + prompt + "\n\n" + context_block + post_prompt


def doc_to_text_with_context_top3(doc: dict, model_specific_kwargs: dict) -> str:
    return _doc_to_text_with_context(doc, model_specific_kwargs, _CONTEXT_TOP3_CSV)


def doc_to_text_with_context_top5(doc: dict, model_specific_kwargs: dict) -> str:
    return _doc_to_text_with_context(doc, model_specific_kwargs, _CONTEXT_TOP5_CSV)


def doc_to_text_with_context_top8(doc: dict, model_specific_kwargs: dict) -> str:
    return _doc_to_text_with_context(doc, model_specific_kwargs, _CONTEXT_TOP8_CSV)


def doc_to_text_multi_round(
    doc: dict,
    model_specific_kwargs: dict,
    round_idx: int | None = None,
    previous_round_results: list | None = None,
    last_round_info: dict | None = None,
) -> str | tuple:
    """Build prompts for multi-round generation used by CIRCLE RAG pipelines."""
    _ = doc
    visual, text = None, None
    should_terminate = False

    if previous_round_results is None:
        previous_round_results = []

    pre_prompt = model_specific_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_kwargs.get("post_prompt", "")
    prompts = model_specific_kwargs.get("prompts")

    if not isinstance(prompts, list) or len(prompts) < 2:
        raise ValueError("`multi_round` expects at least two prompts")

    if round_idx is None:
        return pre_prompt + prompts[0] + post_prompt

    if round_idx < len(prompts):
        text = pre_prompt + prompts[round_idx] + post_prompt
    else:
        should_terminate = True

    return visual, text, should_terminate, previous_round_results, last_round_info


def doc_to_visual(doc: dict) -> list:
    """Convert a TongeDx2 sample to the visual input expected by models."""
    if "visual" in doc:
        image_path = Path(doc["visual"])
    else:
        task_root = Path(__file__).resolve().parents[1]
        image_path = _resolve_image_path(task_root, str(doc["image_path"]))

    return [Image.open(image_path).convert("RGB")]


def _doc_to_visual_with_context(doc: dict, context_csv_name: str) -> list:
    task_root = Path(__file__).resolve().parents[1]
    target_key = _doc_image_key(doc)
    rows = _get_filtered_context_rows(task_root, context_csv_name, target_key)

    visuals = [
        Image.open(_resolve_image_path(task_root, image_rel_path)).convert("RGB")
        for image_rel_path, _ in rows
    ]
    visuals.extend(doc_to_visual(doc))

    if os.getenv("CIRCLE_DEBUG_VISUAL_INPUTS", "0").lower() in {"1", "true", "yes", "on"}:
        if "visual" in doc:
            target_path = str(Path(doc["visual"]))
        else:
            target_path = str(_resolve_image_path(task_root, str(doc["image_path"])))

        print(
            f"[CTXDBG] csv={context_csv_name} doc_visual={doc.get('visual')} "
            f"target_path={target_path} visual_count={len(visuals)} last_image_source={target_path}",
            flush=True,
        )

    return visuals


def doc_to_visual_with_context_top3(doc: dict) -> list:
    return _doc_to_visual_with_context(doc, _CONTEXT_TOP3_CSV)


def doc_to_visual_with_context_top5(doc: dict) -> list:
    return _doc_to_visual_with_context(doc, _CONTEXT_TOP5_CSV)


def doc_to_visual_with_context_top8(doc: dict) -> list:
    return _doc_to_visual_with_context(doc, _CONTEXT_TOP8_CSV)


def doc_to_target(doc: dict) -> str:
    """Return the ground-truth 8-attribute hard-label string."""
    if "target" in doc:
        return str(doc["target"])

    values = [int(doc[attr]) for attr in ATTRIBUTES]
    return " ".join(str(v) for v in values)


def doc_to_target_dummy(doc: dict) -> str:
    """Dummy doc_to_target function that returns an empty string for no evaluation."""
    return ""


def _process_docs_excluding_context(dataset: datasets.Dataset, context_csv_name: str) -> datasets.Dataset:
    task_root = Path(__file__).resolve().parents[1]
    blocked_keys = _context_image_keys(str(task_root), context_csv_name)

    def _keep(example: dict) -> bool:
        return _doc_image_key(example) not in blocked_keys

    return dataset.filter(_keep)


def process_docs_exclude_context_top3(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs_excluding_context(dataset, _CONTEXT_TOP3_CSV)


def process_docs_exclude_context_top5(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs_excluding_context(dataset, _CONTEXT_TOP5_CSV)


def process_docs_exclude_context_top8(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs_excluding_context(dataset, _CONTEXT_TOP8_CSV)


def _build_target_string(row: pd.Series) -> str:
    """Serialize the 8 binary attributes in a fixed order."""
    values = [int(row[attr]) for attr in ATTRIBUTES]
    return " ".join(str(v) for v in values)


def download(output_dir: str = "data", cache_dir: str = ".cache") -> datasets.DatasetDict:
    """Build a local HuggingFace dataset from train_fold1 + val_fold1 CSV files."""
    _ = cache_dir  # Unused, kept for API compatibility with other tasks.

    task_root = Path(__file__).resolve().parents[1]
    list_root = task_root / "release" / "list"

    output_path = Path(output_dir) / "tongedx2"
    if output_path.exists():
        return

    split_files = [list_root / "train_fold1.csv", list_root / "val_fold1.csv"]
    required_cols = ["image_path", *ATTRIBUTES]

    data_frames = []
    for csv_file in split_files:
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing required file: {csv_file}")
        data_frames.append(pd.read_csv(csv_file, usecols=required_cols))

    df = pd.concat(data_frames, ignore_index=True)

    image_index = _build_image_index(str(task_root))
    image_keys = df["image_path"].astype(str).map(_normalize_image_key)
    missing_mask = ~image_keys.isin(image_index.keys())
    if missing_mask.any():
        missing_examples = df.loc[missing_mask, "image_path"].astype(str).head(5).tolist()
        raise FileNotFoundError(
            "Some images listed in CSV are missing from flat root "
            f"'{_image_root(task_root)}'. Examples: {missing_examples}"
        )

    visuals = image_keys.map(image_index).tolist()
    targets = (
        df[ATTRIBUTES]
        .astype("int8")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )

    dataset = datasets.Dataset.from_dict({"visual": visuals, "target": targets})
    data = datasets.DatasetDict({"test": dataset})
    data.save_to_disk(output_path)
