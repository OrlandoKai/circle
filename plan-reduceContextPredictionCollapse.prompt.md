## Plan: Reduce Context Prediction Collapse

Current results show partial recovery (`unique_preds=2`) but strong mode collapse remains, meaning the model still overuses a dominant template output instead of image-specific decisions. The plan is to keep your current pipeline, add a stronger anti-collapse prompt strategy in new task YAMLs, and run a controlled Top3/Top5/Top8 A/B to verify whether diversity and per-attribute F1 improve without changing your dataset split logic.

### Steps 5
1. Quantify collapse by task using single-file logs and compare `unique_preds` plus per-attribute positive rates from [eval_tongedx2_metrics.py](eval_tongedx2_metrics.py).
2. Create new prompt-only task configs beside existing ones, e.g. [src/data/tasks/_classification/TongeDx2/context_top3_single.yaml](src/data/tasks/_classification/TongeDx2/context_top3_single.yaml) and [src/data/tasks/_classification/TongeDx2/context_top8_single.yaml](src/data/tasks/_classification/TongeDx2/context_top8_single.yaml), keeping originals unchanged.
3. Rewrite `prompt` to force target-first visual judgment and reference-as-threshold calibration only, explicitly suppressing label-frequency copying in `model_specific_kwargs.default.prompt`.
4. Add optional context summary text in [src/data/tasks/_classification/TongeDx2/assets/_tongedx2_utils.py](src/data/tasks/_classification/TongeDx2/assets/_tongedx2_utils.py) within `_doc_to_text_with_context` to present per-attribute 0/1 balance cues before output.
5. Run controlled A/B across old vs new prompts for Top3/Top5/Top8 and compare `micro_f1`, `macro_f1`, `sample_accuracy`, and `unique_preds` from identical `--limit` settings.

### Further Considerations 2
1. Prompt strategy choice: Option A strict anti-prior constraints, Option B attribute-by-attribute rubric hints, Option C both combined with shortest output contract.
2. Scope choice: Option A prompt-only YAML changes, Option B prompt + `_doc_to_text_with_context` summary enhancement, Option C also tune generation (`num_beams`, `max_new_tokens`) after prompt A/B stabilizes.

