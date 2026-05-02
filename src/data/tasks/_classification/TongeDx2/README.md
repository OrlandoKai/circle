# TongeDx2 Task

This task evaluates 8 hard-label tongue attributes with merged statistics over `train_fold1 + val_fold1`.
The local dataset is built from CSV files and loaded from disk.

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
- Data source:
  - `release/list/train_fold1.csv`
  - `release/list/val_fold1.csv`
  - images from `release/origin`

## Run Evaluation (Direct Baseline)

```bash
python eval_model.py --model qwen3-vl-8b --tasks tongedx2_direct --log_samples --output_path logs/tongedx2_direct
```

## Run Evaluation (CIRCLE)

```bash
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



