import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ATTRIBUTES = [
    "TonguePale",
    "TipSideRed",
    "Spot",
    "Ecchymosis",
    "Crack",
    "Toothmark",
    "FurThick",
    "FurYellow",
]


def _strict_binary_pattern(expected_len: int) -> re.Pattern:
    if expected_len <= 1:
        return re.compile(r"^[01]$")
    return re.compile(rf"^[01]( [01]){{{expected_len - 1}}}$")


def _parse_binary_vector(text: str, expected_len: int) -> tuple[list[int], bool, str]:
    """Parse a binary vector from free text.

    Returns a tuple of (parsed_vector, is_valid_format, invalid_reason).
    The vector is always resized to expected_len by truncation or zero-padding.
    A valid format is strictly: exactly expected_len values, each 0/1, separated
    by a single ASCII space, with no extra text.
    """
    raw_text = str(text).strip()
    pattern = _strict_binary_pattern(expected_len)
    if pattern.fullmatch(raw_text):
        return [int(x) for x in raw_text.split(" ")], True, ""

    token_matches = re.findall(r"\b[01]\b", raw_text)

    if len(token_matches) < expected_len:
        token_matches = re.findall(r"[01]", raw_text)

    values = [int(x) for x in token_matches[:expected_len]]
    invalid_reason = "invalid_binary_vector_format"
    if re.search(r"[^01\s]", raw_text):
        invalid_reason = "contains_non_binary_content"
    elif len(values) != expected_len:
        invalid_reason = "wrong_binary_value_count"

    if len(values) < expected_len:
        values.extend([0] * (expected_len - len(values)))

    return values, False, invalid_reason


def _write_error_samples(error_samples: list[dict], output_path: str) -> None:
    """Write invalid-format prediction samples to JSONL or CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        pd.DataFrame(error_samples).to_csv(path, index=False)
        return

    with open(path, "w", encoding="utf-8") as f:
        for row in error_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_input_files(input_path: str) -> list[Path]:
    """Resolve input jsonl files from a file/folder/glob argument."""
    if "*" in input_path:
        return sorted(Path().glob(input_path))

    path = Path(input_path)
    if path.is_dir():
        return sorted(path.glob("**/*_samples_*.jsonl"))

    return [path]


def _extract_prediction_text(raw_prediction: object) -> str:
    """Extract a text response from nested per-sample response payloads."""
    current = raw_prediction
    while isinstance(current, list | tuple) and len(current) > 0:
        current = current[-1]
    return str(current)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, attributes: list[str]) -> dict:
    """Compute the requested merged metrics for TongeDx2."""
    def _binary_f1(y_true_1d: np.ndarray, y_pred_1d: np.ndarray) -> float:
        tp = np.sum((y_true_1d == 1) & (y_pred_1d == 1))
        fp = np.sum((y_true_1d == 0) & (y_pred_1d == 1))
        fn = np.sum((y_true_1d == 1) & (y_pred_1d == 0))

        denom = (2 * tp) + fp + fn
        if denom == 0:
            return 0.0

        return float((2 * tp) / denom)

    # Multi-label micro-F1 is computed over all attribute bits at once.
    micro_f1 = _binary_f1(y_true.reshape(-1), y_pred.reshape(-1))

    sample_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    attribute_accuracy = float(np.mean(y_true == y_pred))

    per_attribute = {}
    per_attribute_accuracy = {}
    per_attribute_counts = {}
    per_attribute_count_str = {}
    per_attribute_values = []
    for idx, name in enumerate(attributes):
        f1_value = _binary_f1(y_true[:, idx], y_pred[:, idx])
        attr_matches = y_true[:, idx] == y_pred[:, idx]
        correct_count = int(np.sum(attr_matches))
        total_count = int(attr_matches.shape[0])
        acc_value = float(np.mean(attr_matches))
        per_attribute[name] = f1_value
        per_attribute_accuracy[name] = acc_value
        per_attribute_counts[name] = {
            "correct_count": correct_count,
            "total_count": total_count,
        }
        per_attribute_count_str[name] = f"{correct_count}/{total_count}"
        per_attribute_values.append(f1_value)

    macro_f1 = float(np.mean(per_attribute_values))

    return {
        "num_samples": int(y_true.shape[0]),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "sample_accuracy": sample_accuracy,
        "attribute_accuracy": attribute_accuracy,
        "per_attribute_f1": per_attribute,
        "per_attribute_accuracy": per_attribute_accuracy,
        "per_attribute_counts": per_attribute_counts,
        "per_attribute_count_str": per_attribute_count_str,
    }


def main(args: Namespace) -> None:
    """Aggregate metrics directly on merged train_fold1+val_fold1 predictions."""
    attributes = [x.strip() for x in args.attributes.split(",") if x.strip()]
    expected_len = len(attributes)

    input_files = _resolve_input_files(args.input)
    if len(input_files) == 0:
        raise FileNotFoundError("No input jsonl files were found.")

    y_true_rows = []
    y_pred_rows = []
    invalid_predictions = 0
    error_samples = []

    for input_file in input_files:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        df = pd.read_json(input_file, lines=True)

        if "resps" in df.columns:
            predictions = df["resps"].tolist()
            prediction_source = "resps"
        elif "filtered_resps" in df.columns:
            predictions = df["filtered_resps"].tolist()
            prediction_source = "filtered_resps"
        else:
            raise ValueError(f"Neither 'resps' nor 'filtered_resps' found in {input_file}")

        targets = df["target"].tolist()

        if len(predictions) == 0:
            continue

        parsed_predictions = []
        parsed_targets = []

        for row_index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
            prediction_text = _extract_prediction_text(prediction)

            pred_vec, is_valid, invalid_reason = _parse_binary_vector(
                str(prediction_text), expected_len
            )
            parsed_prediction_vec = pred_vec.copy()
            if not is_valid:
                invalid_predictions += 1

            target_vec, target_ok, target_reason = _parse_binary_vector(str(target), expected_len)
            if not target_ok:
                raise ValueError(
                    f"Target has invalid format in {input_file}: expected {expected_len} values,"
                    f" got '{target}' (reason: {target_reason})"
                )

            if not is_valid:
                # Strict reject: malformed outputs are always counted as fully wrong.
                strict_used_fallback = True
                pred_vec = [1 - v for v in target_vec]

                error_samples.append(
                    {
                        "input_file": str(input_file),
                        "prediction_source": prediction_source,
                        "row_index": row_index,
                        "doc_id": int(df.iloc[row_index]["doc_id"])
                        if "doc_id" in df.columns
                        else None,
                        "raw_prediction": str(prediction_text),
                        "raw_target": str(target),
                        "expected_len": expected_len,
                        "error_type": "invalid_binary_vector_format",
                        "format_error_reason": invalid_reason,
                        "extracted_token_count": int(
                            len(re.findall(r"\b[01]\b", str(prediction_text)))
                        ),
                        "parsed_prediction_vector": parsed_prediction_vec,
                        "used_fallback_vector": pred_vec,
                        "strict_counted_as_wrong": strict_used_fallback,
                    }
                )

            parsed_predictions.append(pred_vec)
            parsed_targets.append(target_vec)

        y_pred_rows.extend(parsed_predictions)
        y_true_rows.extend(parsed_targets)

    if len(y_true_rows) == 0:
        raise ValueError("No valid samples available to compute metrics.")

    y_true = np.asarray(y_true_rows, dtype=np.int32)
    y_pred = np.asarray(y_pred_rows, dtype=np.int32)

    outputs = _compute_metrics(y_true, y_pred, attributes)
    outputs["invalid_prediction_format_count"] = invalid_predictions
    outputs["strict_format"] = True
    outputs["strict_wrong_sample_count"] = len(error_samples)

    print("TongeDx2 merged metrics (train_fold1 + val_fold1):")
    print(f"num_samples: {outputs['num_samples']}")
    print(f"micro_f1: {outputs['micro_f1']:.6f}")
    print(f"macro_f1: {outputs['macro_f1']:.6f}")
    print(f"sample_accuracy: {outputs['sample_accuracy']:.6f}")
    print(f"attribute_accuracy: {outputs['attribute_accuracy']:.6f}")
    print(f"invalid_prediction_format_count: {outputs['invalid_prediction_format_count']}")
    print(f"strict_format: {outputs['strict_format']}")
    print(f"strict_wrong_sample_count: {outputs['strict_wrong_sample_count']}")

    print("per_attribute_f1:")
    for name, value in outputs["per_attribute_f1"].items():
        print(f"  - {name}: {value:.6f}")

    print("per_attribute_accuracy:")
    for name, value in outputs["per_attribute_accuracy"].items():
        counts = outputs["per_attribute_counts"][name]
        print(
            f"  - {name}: {value:.6f} "
            f"({counts['correct_count']}/{counts['total_count']})"
        )

    print("per_attribute_count_str:")
    for name, value in outputs["per_attribute_count_str"].items():
        print(f"  - {name}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)
        print(f"saved_to: {output_path}")

    # Always echo the full metrics JSON so users can inspect it directly in terminal.
    print("metrics_json:")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))

    if args.error_samples_output and len(error_samples) > 0:
        _write_error_samples(error_samples, args.error_samples_output)
        print(f"saved_error_samples_to: {args.error_samples_output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to a jsonl file, folder, or glob for *_samples_*.jsonl files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Optional output path for a JSON summary file.",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        default=",".join(DEFAULT_ATTRIBUTES),
        help="Comma-separated attribute order used for parsing and per-attribute reporting.",
    )
    parser.add_argument(
        "--strict-format",
        action="store_true",
        help=(
            "Deprecated: strict reject is always enabled; malformed predictions are always"
            " counted as fully wrong."
        ),
    )
    parser.add_argument(
        "--error-samples-output",
        type=str,
        default="",
        help=(
            "Optional path to save malformed prediction samples (JSONL by default,"
            " CSV if suffix is .csv)."
        ),
    )

    main(parser.parse_args())




