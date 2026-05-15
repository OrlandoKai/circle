# TongeDx2 Task

This task evaluates 8 hard-label tongue attributes.
The local dataset is built from CSV files and loaded from disk with separate support, validation, and test splits.

## What is added here

This folder includes custom extensions for TongeDx2:

- **Context variants (3/5/8 pairs)**: task configs that inject 3/5/8 reference examples into the prompt and visuals.
- **Single-image only task**: a baseline that removes context and predicts from the target image alone.
- **Utility functions** in `assets/_tongedx2_utils.py` to build prompts, visuals, and filtered datasets for the above.

## Attribute Order

1. TonguePale
2. TipSideRed
3. Spot
4. Ecchymosis
5. Crack
6. Toothmark
7. FurThick
8. FurYellow

## Task Configs

- CIRCLE task: `tongedx2_circle` (`circle.yaml`)
- CIRCLE Chinese prompt variant: `tongedx2_circlechiness` (`circlechiness.yaml`)
- Direct baseline task: `tongedx2_direct` (`direct.yaml`)
- Single-image only: `tongedx2_single_image_only` (`single_image_only.yaml`)
- Context top-3: `tongedx2_context_top3_single` (`context_top3_single.yaml`)
- Context top-5: `tongedx2_context_top5_single` (`context_top5_single.yaml`)
- Context top-8: `tongedx2_context_top8_single` (`context_top8_single.yaml`)
- Data source:
  - `release/list/train_fold1.csv` -> `support` / `train` split for retrieval
  - `release/list/val_fold1.csv` -> `val` split for prompt and hyperparameter checks
  - `release/list/test.csv` -> `test` split for final evaluation
  - images from `release/origin`

## Utilities in `assets/_tongedx2_utils.py`

Key helpers used by the configs:

- `doc_to_text` / `doc_to_visual` / `doc_to_target`: build the prompt, visual input, and ground-truth target.
- `doc_to_text_with_context_top3|5|8`: inserts reference labels into the prompt.
- `doc_to_visual_with_context_top3|5|8`: prepends reference images to the visual list.
- `process_docs_exclude_context_top3|5|8`: filters out target images that appear in the context CSVs.
- `download`: builds a local HF dataset with `support`, `train`, `val`, and `test` splits.

The context CSVs are:
- `train_fold1_min_top3_exact_AM.csv`
- `train_fold1_min_top5_exact_AM.csv`
- `train_fold1_min_top8_exact_AM.csv`

## Run Evaluation (Direct Baseline)

```bash
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_direct --log_samples --output_path logs/tongedx2_direct
```

## Run Evaluation (Single-Image Only)

```bash
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_single_image_only --log_samples --output_path logs/tongedx2_single_image_only
```

## Run Evaluation (Context 3/5/8)

```bash
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_context_top3_single --log_samples --output_path logs/tongedx2_context_top3
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_context_top5_single --log_samples --output_path logs/tongedx2_context_top5
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_context_top8_single --log_samples --output_path logs/tongedx2_context_top8
```

## Run Evaluation (CIRCLE)

```bash
python eval_model.py --model clip-vit-b32-openai --tasks tongedx2_embed_support --batch_size 128
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_circle --log_samples --output_path logs/tongedx2_circle
```

## Compute Merged Metrics

```bash
python eval_tongedx2_metrics.py -i logs/tongedx2_circle -o logs/tongedx2_circle/tongedx2_metrics.json
```

## Compute Merged Metrics (Strict Format)

Malformed predictions are strictly rejected during scoring (counted as fully wrong),
and an error sample list can be exported.

```bash
python eval_tongedx2_metrics.py -i logs/tongedx2_circle --error-samples-output logs/tongedx2_circle/tongedx2_invalid_samples.jsonl -o logs/tongedx2_circle/tongedx2_metrics_strict.json
```

The strict-mode error list includes both:
- `parsed_prediction_vector`: vector parsed from the malformed text before strict fallback
- `used_fallback_vector`: vector actually used for scoring in strict mode

The script reports:
- MicroF1
- MacroF1
- Sample accuracy (all 8 bits must match)
- Attribute accuracy (bit-level average)
- Per-attribute F1 (each of the 8 attributes)
- Per-attribute accuracy (each of the 8 attributes)
- Per-attribute correct/total counts (for example, `37/60`)
- Flat map output key: `per_attribute_count_str` (for example, `"TonguePale": "37/60"`)
- Invalid prediction format count

