import base64
import copy
import os
from collections.abc import Iterable
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers.utils import ModelOutput

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

__all__ = ["Qwen2VL"]

log = utils.get_logger(__name__, rank_zero_only=True)


def _flatten_list(input: list[list[Any]]) -> list[Any]:
    """Flatten a nested list into a single list.

    Args:
    ----
        input (list): A nested list containing elements to be flattened.

    """
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def _iter_images(visual: Any) -> Iterable[Image.Image]:
    """Yield one or more PIL images from a visual payload."""
    if isinstance(visual, Image.Image):
        yield visual
        return

    if isinstance(visual, list | tuple):
        for item in visual:
            if isinstance(item, Image.Image):
                yield item


def _get_primary_image(visual: Any) -> Image.Image | None:
    """Pick a stable representative image for retrieval-oriented paths."""
    for image in _iter_images(visual):
        return image
    return None


def _describe_visual_item_for_debug(item: Any) -> str:
    """Return a short, stable descriptor for a visual item."""
    if isinstance(item, Image.Image):
        return f"PIL(size={item.size}, mode={item.mode})"
    if isinstance(item, str):
        return item
    if item is None:
        return "None"
    return type(item).__name__


def _summarize_visual_for_debug(visual: Any) -> dict[str, Any]:
    """Summarize one visual payload for debug logging."""
    if isinstance(visual, list | tuple):
        if len(visual) == 0:
            return {"count": 0, "kind": type(visual).__name__, "first": "None", "last": "None"}
        return {
            "count": len(visual),
            "kind": type(visual).__name__,
            "first": _describe_visual_item_for_debug(visual[0]),
            "last": _describe_visual_item_for_debug(visual[-1]),
        }
    return {
        "count": 0 if visual is None else 1,
        "kind": type(visual).__name__,
        "first": _describe_visual_item_for_debug(visual),
        "last": _describe_visual_item_for_debug(visual),
    }


class Qwen2VL(Model):
    """Qwen2VL Model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "Qwen/Qwen2-VL-7B-Instruct".
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether to use flash attention 2. Default to False.
        max_pixels (int): The max number of pixels in an image. Defaults to 12'845'056.
        min_pixels (int): The min number of pixels in an image. Defaults to 3'316'.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://github.com/QwenLM/Qwen2-VL

    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_cache: bool = True,
        use_flash_attention_2: bool | None = utils.package_available("flash_attn"),
        max_pixels: int = 1024 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._use_cache = use_cache
        self._use_flash_attention_2 = use_flash_attention_2
        self._max_pixels = max_pixels
        self._min_pixels = min_pixels
        self.batch_size_per_gpu = batch_size

        if device_map == "None":
            device_map = None

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
        }
        processor_kwargs = {
            "max_pixels": self._max_pixels,
            "min_pixels": self._min_pixels,
        }

        if self._use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        PretrainedModel = Qwen2VLForConditionalGeneration
        if "Qwen3" in self._model_name_or_path:
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError as e:
                raise ValueError(
                    "Failed to import Qwen3VLForConditionalGeneration."
                    " Please upgrade transformers to a later version."
                ) from e

            PretrainedModel = Qwen3VLForConditionalGeneration
        elif "Qwen2.5" in self._model_name_or_path:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
            except ImportError as e:
                raise ValueError(
                    "Failed to import Qwen2_5_VLForConditionalGeneration."
                    " Please upgrade transformers to a later version."
                ) from e

            PretrainedModel = Qwen2_5_VLForConditionalGeneration

        self._model = PretrainedModel.from_pretrained(self._model_name_or_path, **model_kwargs)
        self._model = torch.compile(self._model, mode="max-autotune", fullgraph=True)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        if "Qwen2.5" in self._model_name_or_path or "Qwen3" in self._model_name_or_path:
            self._processor.tokenizer.padding_side = "left"
            self._tokenizer.padding_side = "left"

    def loglikelihood(self, requests: list[TaskInstance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.

        Downstream tasks should attempt to use loglikelihood instead of other
        LMM calls whenever possible.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, continuation). The arguments are as follows:
                - context (str): Context string. Implementations of LMM must be able to handle an
                    empty context string.
                - continuation (str):  The continuation over which log likelihood will be
                    calculated. If there is a word boundary, the space should be in the
                    continuation, e.g., context="hello" continuation=" world" is correct.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError

    def _loglikelihood(
        self, inputs: dict, generation_output: ModelOutput
    ) -> dict[str, torch.Tensor | list]:
        """Compute log-likelihood and related metrics for generated sequences.

        Args:
        ----
            inputs: The input tensors used for generation.
            generation_output: The output from the model's generate method.

        """
        cont = generation_output.sequences
        scores = generation_output.scores

        transition_scores = self.model.compute_transition_scores(
            cont,
            scores,
            normalize_logits=True,  # ensure proper log probabilities (softmax is applied)
        )

        # Extract generated tokens (excluding prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = cont[:, input_length:]  # (batch_size, max_new_tokens)

        metrics = {
            "loglikelihood": [],
            "likelihood": [],
            "avg_loglikelihood": [],
            "per_token_loglikelihood": [],
            "perplexity": [],
        }

        for idx in range(generated_tokens.shape[0]):
            gen_seq = generated_tokens[idx]
            trans_scores = transition_scores[idx]  # Log probs for this sequence

            # Find position of first EOS (include it in the sum)
            eos_positions = (gen_seq == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                eff_length = eos_positions[0].item() + 1
            else:
                eff_length = gen_seq.shape[0]  # Full length if no EOS

            # Sequence metrics
            sequence_log_prob = trans_scores[:eff_length].sum()
            sequence_prob = torch.exp(sequence_log_prob)  # May be very small
            avg_log_prob = sequence_log_prob / eff_length if eff_length > 0 else torch.tensor(0.0)
            perplexity = torch.exp(-avg_log_prob) if eff_length > 0 else torch.tensor(float("inf"))

            metrics["loglikelihood"].append(sequence_log_prob)
            metrics["likelihood"].append(sequence_prob)
            metrics["avg_loglikelihood"].append(avg_log_prob)
            metrics["per_token_loglikelihood"].append(trans_scores[:eff_length])
            metrics["perplexity"].append(perplexity)

        for k in ["loglikelihood", "likelihood", "avg_loglikelihood", "perplexity"]:
            metrics[k] = torch.stack(metrics[k])

        return metrics

    def _is_rag_enabled(self, gen_kwargs: dict) -> bool:
        """Check if RAG is enabled in generation kwargs.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.

        """
        rag = (gen_kwargs or {}).get("rag") or {}
        return rag.get("enabled", False)

    def _setup_rag(self, gen_kwargs: dict, multi_step: bool = False) -> None:
        """Set up RAG retriever if not already set up.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.
            multi_step (bool): Whether to set up multi-step RAG retrievers. Defaults to False.

        """
        if hasattr(self, "_retriever"):
            return

        db_root = os.getenv("RAG_DATABASE_ROOT")
        if not db_root:
            log.warning(
                "RAG is enabled but RAG_DATABASE_ROOT is not set; skipping retriever init."
            )
            return

        rag = (gen_kwargs or {}).get("rag") or {}

        if not multi_step:
            self._retriever = Retriever(
                Path(db_root) / rag.get("database_path"),
                format=rag.get("database_format", "faiss"),
                model_name=rag.get("model_name"),
                few_shot=rag.get("num_shots_per_class"),
            )
            self._retriever.set_vocab_transform(rag)

        else:
            self._retriever = {}
            configs = {}
            for step_idx, rag_config in rag.items():
                # Not actual RAG, just piping from previous step
                if rag_config.get("pipe_from") is not None:
                    continue

                db_path = rag_config.get("database_path")
                # Reuse retriever if already loaded
                if db_path in configs:
                    self._retriever[step_idx] = self._retriever[configs[db_path]]
                else:
                    self._retriever[step_idx] = Retriever(
                        Path(db_root) / rag_config.get("database_path"),
                        format=rag_config.get("database_format", "faiss"),
                        model_name=rag_config.get("model_name"),
                        few_shot=rag_config.get("num_shots_per_class"),
                    )
                    configs[db_path] = step_idx
                self._retriever[step_idx].set_vocab_transform(rag_config)

        log.info("Retrieval database(s) loaded!")

    def _retrieve(
        self,
        gen_kwargs: dict,
        images: list[Image.Image],
        doc_ids: dict | None = None,
        step_idx: int | None = None,
    ) -> None | list[list[dict[str, Any]]]:
        """Retrieve relevant data for each image using the retriever.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.
            images (list[Image.Image]): List of images to retrieve data for.
            doc_ids (dict | None): Optional dictionary mapping document IDs to their metadata.
            step_idx (int | None): Optional index for multi-step RAG retrieval.

        """
        if not hasattr(self, "_retriever"):
            log.error("RAG is enabled but retriever is not set up.")
            exit()

        def _prepare(image: Image.Image, resize_image_size: None | int = None) -> dict[str, Any]:
            base64_image = image.convert("RGB")

            if resize_image_size is not None and max(base64_image.size) > resize_image_size:
                base64_image.thumbnail((resize_image_size, resize_image_size), Image.LANCZOS)

            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")

            payload = {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}

            return payload

        prepare = partial(
            _prepare,
            resize_image_size=(gen_kwargs or {}).get("rag", {}).get("resize_images", None),
        )

        use_custom_prepare = not (gen_kwargs or {}).get("rag", {}).get("separate_queries", False)

        retriever = self._retriever
        if step_idx is not None:
            retriever = retriever[step_idx]
        result = retriever.retrieve_and_prepare(
            gen_kwargs,
            images,
            prepare=prepare if use_custom_prepare else lambda x: x,
            doc_ids=doc_ids,
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        result = cast(list[list[dict[str, Any]]], result)

        return result

    def _batch_rag(
        self,
        batched_doc_id: Iterable,
        batched_contexts: Iterable,
        batched_visuals: Iterable,
        gen_kwargs: dict,
        task: str,
        split: str,
        step_idx: int | None = None,
    ) -> tuple[dict, dict[int, list[str]], dict[int, list[str]]]:
        """Retrieve data for a batch of samples using RAG.

        Args:
        ----
            batched_doc_id (Iterable): An iterable of document IDs for the batch.
            batched_contexts (Iterable): An iterable of context strings for the batch.
            batched_visuals (Iterable): An iterable of visual inputs for the batch.
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            task (str): The task name.
            split (str): The data split name.
            step_idx (int | None): Optional index for multi-step RAG retrieval.

        """
        # RAG optimization: pre-load all images and retrieve in batches
        rag_messages = {}

        cache_and_reuse = (gen_kwargs or {}).get("rag", {}).get("cache_and_reuse", False)
        reuse = (gen_kwargs or {}).get("rag", {}).get("reuse", False)

        batched_visuals_for_rag = {}
        for i, _ in enumerate(batched_contexts):
            visual = batched_visuals[i] if i < len(batched_visuals) else None
            visual = _get_primary_image(visual)
            if isinstance(visual, Image.Image):
                visual = visual.convert("RGB")
                batched_visuals_for_rag[i] = visual

                # When caching and reusing, only retrieve for the first image
                # Typically, this is used with random sampling, so there's no
                # relationship between the images and the retrieved content,
                # meaning we can just retrieve once and reuse the same content for all
                if cache_and_reuse or reuse:
                    break

        # Batch retrieve data
        payload = [{"doc_id": ids, **self.task_dict[task][split][ids]} for ids in batched_doc_id]
        retrieved_data = self._retrieve(
            gen_kwargs, list(batched_visuals_for_rag.values()), payload, step_idx=step_idx
        )
        for k, v in zip(batched_visuals_for_rag.keys(), retrieved_data, strict=True):
            rag_messages[k] = v

        retrieved_filenames: dict[int, list[str]] = {}
        retrieved_targets: dict[int, list[str]] = {}
        retriever = self._retriever[step_idx] if step_idx is not None else self._retriever
        if hasattr(retriever, "get_last_retrieved_filenames"):
            filename_groups = retriever.get_last_retrieved_filenames()
            for key, names in zip(batched_visuals_for_rag.keys(), filename_groups, strict=True):
                retrieved_filenames[key] = names

        if hasattr(retriever, "get_last_retrieved_targets"):
            target_groups = retriever.get_last_retrieved_targets()
            for key, targets in zip(batched_visuals_for_rag.keys(), target_groups, strict=True):
                retrieved_targets[key] = targets

        return rag_messages, retrieved_filenames, retrieved_targets

    def _prepare_visual_for_context(self, visual: Image.Image) -> str:
        """Prepare a visual input for context by converting it to a base64-encoded string.

        Args:
        ----
            visual (Image.Image): The visual input to prepare.

        """
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        return base64_image, base64_string

    def _make_history(
        self,
        gen_kwargs: dict,
        batched_contexts: Iterable,
        batched_visuals: Iterable,
        rag: dict,
        rag_messages: dict,
        history: list | None = None,
        append_history: bool = False,
        step_idx: int | None = None,
    ) -> tuple[list, list]:
        """Construct the conversation history for each sample in the batch.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            batched_contexts (Iterable): An iterable of context strings for the batch.
            batched_visuals (Iterable): An iterable of visual inputs for the batch.
            rag (dict): RAG configuration dictionary.
            rag_messages (dict): A dictionary mapping from sample index to retrieved RAG messages.
            history (list | None): Optional list of conversation histories for each sample.
                If None, a default history will be created for each sample. Defaults to None.
            append_history (bool): Whether to append the provided history to the default history
                (True) or to replace it entirely (False). Defaults to False.
            step_idx (int | None): Optional index for multi-step RAG retrieval, used to determine
                which retrieved messages to include in the history.

        """
        if history is not None:
            assert len(history) == len(
                batched_contexts
            ), "There must be a conversation for each sample (context)"

        rag_enabled = rag["rag_enabled"]
        rag_position = rag.get("position", "pre-sample")
        include_target_classes = rag.get("include_target_classes") is not None

        if include_target_classes:
            include_target_classes_position = rag.get("include_target_classes", "pre-query")
            target_classes_format = rag.get("target_classes_format", "csv")
            target_classes = sorted(set([rag["doc_to_target"](x) for x in rag["test_docs"]()]))
            target_classes_prompt = rag.get("target_classes_prompt")
            if target_classes_format == "csv":
                target_classes_str = ",".join(target_classes)
            elif target_classes_format == "csv-spaces":
                target_classes_str = ", ".join(target_classes)
            elif target_classes_format == "newline":
                target_classes_str = "\n".join(target_classes)
            elif target_classes_format == "bullet-list":
                target_classes_str = "\n".join([f"- {cls}" for cls in target_classes])
            elif target_classes_format == "numbered-list":
                target_classes_str = "\n".join(
                    [f"{i+1}. {cls}" for i, cls in enumerate(target_classes)]
                )
            else:
                raise ValueError(f"Unknown target_classes_format: {target_classes_format}")

        messages = []
        images_per_request = []

        for i, context in enumerate(batched_contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            # Use a default message as initial context
            # If history is provided, replace the message with the history for the i-th sample
            # `history` should contain the system prompt, as it entirely replaces the default
            # message -- it is not appended to it!
            message = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                    if rag.get("system_prompt") is None
                    else rag.get("system_prompt"),
                }
            ]
            if history is not None and history[i] is not None:
                if append_history:
                    message.extend(history[i])
                else:
                    message = history[i]

            if len(batched_visuals) > 0:
                visual = batched_visuals[i] if i < len(batched_visuals) else None
                _images_counter = len(visual) if isinstance(visual, list | tuple) else 1
                if isinstance(visual, Image.Image):  # Single image
                    base64_image, base64_string = self._prepare_visual_for_context(visual)

                    rag_message = None
                    if rag_enabled:
                        if i in rag_messages:
                            rag_message = rag_messages[i]
                        else:
                            rag_message = self._retrieve(
                                gen_kwargs, [base64_image], step_idx=step_idx
                            )[0]

                        # Update the image counter, used for stats
                        _images_counter += sum(
                            1
                            for _msg in rag_message
                            for k, v in _msg.items()
                            if k == "type" and v == "image"
                        )

                    # Construct the message
                    content = []

                    if (
                        include_target_classes
                        and include_target_classes_position == "begin-of-ctx"
                    ):
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    # Retrieved data goes at the beginning of the context
                    if rag_enabled and rag_position == "pre-sample" and rag_message is not None:
                        content.extend(rag_message)

                    if rag.get("pre_image_prompt") is not None:
                        content.append({"type": "text", "text": rag["pre_image_prompt"]})

                    # When RAG is disabled, always include the image
                    # When it is enabled, check the "include_image" flag (True by default)
                    if not rag_enabled or rag.get("include_image", True):
                        content.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )

                    if include_target_classes and include_target_classes_position == "after-image":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    # Retrieved data goes after the image, before the query
                    if rag_enabled and rag_position == "post-sample" and rag_message is not None:
                        content.extend(rag_message)

                    if include_target_classes and include_target_classes_position == "pre-query":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    content.append({"type": "text", "text": context})

                    # Retrieved data goes at the end of the context
                    if (
                        rag_enabled
                        and rag_position == "post-sample-and-query"
                        and rag_message is not None
                    ):
                        content.extend(rag_message)

                    if include_target_classes and include_target_classes_position == "end-of-ctx":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    message.append(
                        {
                            "role": "user",
                            "content": content,
                        }
                    )
                    images_per_request.append(_images_counter)

                elif isinstance(visual, list | tuple) and all(
                    isinstance(v, Image.Image) for v in visual
                ):  # Multiple images
                    image_content = []
                    for v in visual:
                        base64_image, base64_string = self._prepare_visual_for_context(v)
                        image_content.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )
                    message.append(
                        {
                            "role": "user",
                            "content": image_content + [{"type": "text", "text": context}],
                        }
                    )
                else:
                    content = []

                    if include_target_classes and include_target_classes_position == "pre-query":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    content.append({"type": "text", "text": context})

                    if include_target_classes and include_target_classes_position == "end-of-ctx":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    message.append({"role": "user", "content": content})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        return messages, images_per_request

    def _generate(self, messages: list, gen_kwargs: dict, rag: dict) -> tuple[dict, ModelOutput]:
        """Generate model outputs for the given messages.

        Args:
        ----
            messages (list): A list of messages.
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            rag (dict): RAG configuration dictionary.

        """
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        if video_inputs is not None:
            raise ValueError("Video inputs should be empty for current implementation of Qwen2VL.")

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda") if self.device_map == "auto" else inputs.to(self.device)
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 128
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        if "resize_input_image" in gen_kwargs:
            gen_kwargs.pop("resize_input_image")

        pad_token_id = self.tokenizer.pad_token_id

        do_sample = gen_kwargs["temperature"] > 0
        generation_config = copy.deepcopy(self.model.generation_config)
        if not do_sample:
            # Prevent Transformers from warning about unused sampling-only flags in greedy mode.
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        model_generate_kwargs = {
            **inputs,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "do_sample": do_sample,
            "num_beams": gen_kwargs["num_beams"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "use_cache": self._use_cache,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_logits": True,
            "generation_config": generation_config,
        }
        if do_sample:
            model_generate_kwargs["temperature"] = gen_kwargs["temperature"]
            if gen_kwargs["top_p"] is not None:
                model_generate_kwargs["top_p"] = gen_kwargs["top_p"]

        generation_output = self.model.generate(**model_generate_kwargs)

        return inputs, generation_output

    def _decode(self, inputs: dict, generation_output: ModelOutput) -> list[str]:
        """Decode the generated output into a list of strings.

        Args:
        ----
            inputs (dict): The input dictionary containing input IDs.
            generation_output (ModelOutput): The output from the model generation.

        """
        cont = generation_output.sequences

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont, strict=True)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return answers

    def _downscale_images(self, images: list[Image.Image], rag: dict) -> list[Image.Image]:
        """Downscale images based on RAG configuration.

        Args:
        ----
            images (list[Image.Image]): List of images to potentially downscale.
            rag (dict): RAG configuration dictionary containing resizing parameters.

        """
        resize_image_size = rag.get("resize_images")
        if resize_image_size is not None:
            for image in images:
                if not isinstance(image, Image.Image):
                    continue

                # Scale down, keeping aspect ratio, if the image is larger
                if max(image.size) > resize_image_size:
                    image.thumbnail((resize_image_size, resize_image_size), Image.LANCZOS)

    def _print_last_layer_probs(
        self,
        generation_output: ModelOutput,
        topk: int,
        max_samples: int,
        task: str | None = None,
        split: str | None = None,
        doc_ids: list[int] | None = None,
    ) -> list[list[dict[str, float | str]] | None] | None:
        scores = getattr(generation_output, "scores", None)
        logits = getattr(generation_output, "logits", None)
        last_step = None
        if scores:
            last_step = scores[-1]
        elif logits is not None:
            last_step = logits[:, -1, :]
        if last_step is None:
            return None

        probs = torch.softmax(last_step, dim=-1)
        k = max(1, min(int(topk), probs.shape[-1]))
        top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

        topk_data: list[list[dict[str, float | str]]] = []
        for idx in range(top_probs.shape[0]):
            token_ids = top_ids[idx].detach().cpu().tolist()
            token_probs = top_probs[idx].detach().cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            topk_data.append(
                [
                    {"token": tok, "prob": float(prob)}
                    for tok, prob in zip(tokens, token_probs, strict=True)
                ]
            )

        max_rows = min(int(max_samples), len(topk_data)) if max_samples is not None else len(topk_data)
        for idx in range(max_rows):
            tag = (
                f"task={task} split={split} doc_id={doc_ids[idx]}"
                if doc_ids and idx < len(doc_ids)
                else f"task={task} split={split}"
            )
            pairs = ", ".join(
                f"{item['token']}:{item['prob']:.6f}" for item in topk_data[idx]
            )
            print(f"[LAST_LAYER] {tag} -> {pairs}", flush=True)

        if max_rows < len(topk_data):
            for idx in range(max_rows, len(topk_data)):
                topk_data[idx] = None

        return topk_data

    def generate_until(self, requests: list[TaskInstance]) -> list[str]:
        """Generate greedily until a stopping sequence.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, until). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        res = []

        def _collate(x: tuple[str, ...]) -> tuple[int, str]:
            """Group and sort requests by context length for efficient batching.

            The negative sign on len(tokens) sorts in descending order, which provides several
                advantages:
                - Time estimates will be overestimates rather than underestimates, which is more
                    useful for planning;
                - The first item in a batch determines the padded context length, simplifying
                    batching logic;
                - Makes automatic adaptive batches much easier to implement;
                - Any out-of-memory errors occur immediately rather than near the end.

            Args:
            ----
                x: A tuple containing the context string and other arguments

            """
            tokens = self.tokenizer.encode(x[0])
            return -len(tokens), x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        debug_visual_inputs = os.getenv("CIRCLE_DEBUG_VISUAL_INPUTS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        debug_visual_max_samples = int(os.getenv("CIRCLE_DEBUG_VISUAL_MAX_SAMPLES", "3"))
        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk, strict=True)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [
                batched_doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in batched_doc_id
            ]
            # Keep one visual payload per sample; do not flatten multi-image contexts.

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            if gen_kwargs.get("resize_input_image", None) is not None:
                for visual in batched_visuals:
                    for image in _iter_images(visual):
                        # Scale down, keeping aspect ratio, if the image is larger
                        if max(image.size) > gen_kwargs["resize_input_image"]:
                            image.thumbnail(
                                (gen_kwargs["resize_input_image"], gen_kwargs["resize_input_image"]),
                                Image.LANCZOS,
                            )

            if debug_visual_inputs:
                for idx, (doc_id, visual) in enumerate(zip(batched_doc_id, batched_visuals, strict=True)):
                    if idx >= debug_visual_max_samples:
                        break
                    doc = self.task_dict[task][split][doc_id]
                    target_visual = doc.get("visual") if isinstance(doc, dict) else None
                    visual_summary = _summarize_visual_for_debug(visual)
                    print(
                        f"[VISDBG] task={task} split={split} doc_id={doc_id} "
                        f"target_visual={target_visual} visual_count={visual_summary['count']} "
                        f"visual_kind={visual_summary['kind']} first={visual_summary['first']} "
                        f"last={visual_summary['last']}",
                        flush=True,
                    )

            # RAG setup (guard-style, avoids deep nesting later)
            rag_enabled = self._is_rag_enabled(gen_kwargs)
            rag = (gen_kwargs or {}).get("rag") or {}
            rag["rag_enabled"] = rag_enabled
            if rag_enabled:
                rag["doc_to_target"] = configurable_task.doc_to_target
                rag["doc_to_visual"] = configurable_task.doc_to_visual
                rag["test_docs"] = configurable_task.test_docs
                self._setup_rag(gen_kwargs)

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        "Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            if isinstance(batched_contexts, tuple):
                batched_contexts = list(batched_contexts)

            for i in range(len(batched_contexts)):
                if "<image>" in batched_contexts[i]:
                    batched_contexts[i] = batched_contexts[i].replace("<image>", "")

            rag_messages = {}
            rag_retrieved_filenames: dict[int, list[str]] = {}
            rag_retrieved_targets: dict[int, list[str]] = {}
            if rag_enabled:
                rag_messages, rag_retrieved_filenames, rag_retrieved_targets = self._batch_rag(
                    batched_doc_id, batched_contexts, batched_visuals, gen_kwargs, task, split
                )

            random_sample_filenames_by_ctx = [
                rag_retrieved_filenames.get(i, rag_retrieved_filenames.get(0, []))
                for i in range(len(batched_contexts))
            ]
            random_sample_targets_by_ctx = [
                rag_retrieved_targets.get(i, rag_retrieved_targets.get(0, []))
                for i in range(len(batched_contexts))
            ]

            messages, images_per_request = self._make_history(
                gen_kwargs,
                batched_contexts,
                batched_visuals,
                rag,
                rag_messages,
            )

            if debug_visual_inputs:
                for idx, count in enumerate(images_per_request[:debug_visual_max_samples]):
                    print(
                        f"[VISDBG] request_idx={idx} images_per_request={count}",
                        flush=True,
                    )

            inputs, generation_output = self._generate(
                messages, copy.deepcopy(gen_kwargs), copy.deepcopy(rag)
            )
            last_layer_topk = None
            if gen_kwargs.get("print_last_layer_probs", False):
                last_layer_topk = self._print_last_layer_probs(
                    generation_output,
                    topk=gen_kwargs.get("last_layer_probs_topk", 5),
                    max_samples=gen_kwargs.get("last_layer_probs_max_samples", 3),
                    task=task,
                    split=split,
                    doc_ids=list(batched_doc_id),
                )
            answers = self._decode(inputs, generation_output)

            for ans_idx, answer in enumerate(answers):
                if "Answer:" in answer:
                    answers[ans_idx] = (
                        answer.split("Answer:")[-1]
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .strip()
                    )

            gen_metrics = self._loglikelihood(inputs, generation_output)

            rag.pop("doc_to_target", None)
            rag.pop("doc_to_visual", None)
            rag.pop("test_docs", None)

            if rag.get("store_memory", False):
                self._retriever.store_memory(batched_visuals, answers)

            for idx, (ans, context) in enumerate(zip(answers, batched_contexts, strict=True)):
                _ans = TaskSingleOutput(
                    answer=ans,
                    last_layer_topk_probs=last_layer_topk[idx] if last_layer_topk else None,
                    context=context,
                    context_tokens_count=inputs["attention_mask"][idx].sum(),
                    num_images=images_per_request[idx] if len(images_per_request) > idx else None,
                    loglikelihood=gen_metrics["loglikelihood"][idx].item(),
                    perplexity=gen_metrics["perplexity"][idx].item(),
                    rag_sample_filenames=random_sample_filenames_by_ctx[idx],
                    rag_sample_targets=random_sample_targets_by_ctx[idx],
                )
                res.append(_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res

    def _pipe_from(
        self,
        rag: dict,
        batched_contexts: list,
        round_idx: int,
        round_idx_to_actual_step: dict,
        answers_by_round: dict,
        data_by_round: list,
        prompts_by_round: list,
    ) -> list:
        """Get data from a previous step.

        Args:
        ----
            rag (dict): RAG configuration dictionary containing piping configuration.
            batched_contexts (list): A list of context strings for the batch.
            round_idx (int): The current round index.
            round_idx_to_actual_step (dict): Determine the actual step to pipe data from.
            answers_by_round (dict): Answers from previous rounds.
            data_by_round (list): Retrieved data from previous rounds.
            prompts_by_round (list): Prompts used in previous rounds.

        """

        def _get_conv(text: str, prompt: str) -> dict[str, str]:
            """Construct a conversation message from text and a prompt.

            Args:
            ----
                text (str): The text to include in the message.
                prompt (str): The prompt template, with a `{}` placeholder for the text.

            """
            return [
                {
                    "type": "text",
                    "text": prompt.format(text),
                }
            ]

        format = rag.get("format", "list-captions")
        prompt = rag.get("prompt", "{}")
        prompts = prompt.split("<images>")

        # When caching and reusing the retrieved data, we only need to
        # process the first item in the batch, and then reuse the same
        # data for all items in the batch
        _iterator = (
            range(len(batched_contexts))
            if (not rag.get("cache_and_reuse", False) and not rag.get("reuse", False))
            else range(1)
        )
        retrieved_data = []
        for ctx_id in _iterator:
            remove_ids = []
            if rag.get("keep_full_history", False):
                if rag.get("circular", False):
                    indices_of_circular = [
                        k
                        for k, v in round_idx_to_actual_step.items()
                        if v == round_idx_to_actual_step.get(rag.get("pipe_from")) and k < round_idx
                    ]
                    indices_of_circular = [rag.get("pipe_from__first_step")] + indices_of_circular

                else:
                    indices_of_circular = [
                        k
                        for k, v in round_idx_to_actual_step.items()
                        if v == round_idx_to_actual_step.get(rag.get("pipe_from"))
                        and k < round_idx
                    ]

                    if rag.get("remove_when_labels_change", False):
                        indices_of_circular = [indices_of_circular[0] - 1] + indices_of_circular

                indices_of_circular = sorted(list(set(indices_of_circular)))
                text = []
                for idx in indices_of_circular:
                    _t = answers_by_round[idx]
                    if isinstance(_t[0], list):
                        _t = _t[ctx_id]
                    text.append(_t)
                text = list(zip(*text, strict=True))

                if len(rag.get("concatenate_labels", "")) > 0:
                    text = [
                        rag.get("concatenate_labels", "").join([t_item.strip() for t_item in t])
                        for t in text
                    ]

                elif rag.get("remove_when_labels_change", False):
                    for t_idx, t in enumerate(text):
                        # Check if all elements in `t` are the same
                        first_elem = t[0].strip()
                        all_same = all(first_elem == t_item.strip() for t_item in t)
                        if not all_same:
                            remove_ids.append(t_idx)

                    # And keep only the last label
                    text = [t[-1] for t in text]

            else:
                text = answers_by_round[rag.get("pipe_from")]

                # In case text is a list of lists (e.g., multiple answers for each item
                # in the batch), select the right one
                if isinstance(text[0], list):
                    text = text[ctx_id]

            if rag.get("circular", False):
                _from_step = rag.get("pipe_from__first_step", rag.get("pipe_from"))
                _data = data_by_round[_from_step]
            elif rag.get("include_image", False):
                _data = data_by_round[
                    0
                ]  # from first step, i.e., the raw images to re-build the history
            else:
                _data = data_by_round[rag.get("pipe_from")]

            if isinstance(_data[0], list):
                _data = _data[ctx_id]
                self._downscale_images(_data, rag)

            if rag.get("exclude_rag_message") is not None:
                text.pop(rag.get("exclude_rag_message"))
                _data.pop(rag.get("exclude_rag_message"))

            # Remove items where labels changed
            _data = copy.deepcopy(_data)
            for remove_id in sorted(remove_ids, reverse=True):
                text.pop(remove_id)
                _data.pop(remove_id)

            if format == "list-captions":
                text = [f"{idx + 1}. {t.strip()}" for idx, t in enumerate(text)]
                text = "\n".join(text)
                _rag_data = _get_conv(text, prompt)

            elif format == "bullet-list-captions":
                text = [f"- {t.strip()}" for t in text]
                text = "\n".join(text)
                _rag_data = _get_conv(text, prompt)

            elif format == "newlines":
                text = "\n".join([t.strip() for t in text])
                _rag_data = _get_conv(text, prompt)

            elif format == "csv":
                text = ", ".join([t.strip() for t in text])
                _rag_data = _get_conv(text, prompt)

            elif format == "interleaved":
                _rag_data = [
                    {"type": "text", "text": prompts[0]},
                ]
                for _image, generated_label in zip(_data, text, strict=True):
                    if isinstance(_image, str):
                        # Already a base64 string
                        base64_string = _image
                    else:
                        _, base64_string = self._prepare_visual_for_context(_image)
                    _rag_data.append(
                        {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}
                    )
                    _rag_data.append({"type": "text", "text": prompts[-1].format(generated_label)})

                if rag.get("end_context_prompt") is not None:
                    _rag_data.append({"type": "text", "text": rag["end_context_prompt"]})

            elif format == "image-answer-pairs":
                _rag_data = []
                for _image, generated_label in zip(_data, text, strict=True):
                    _, base64_string = self._prepare_visual_for_context(_image)
                    _res = []

                    _res.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompts_by_round[
                                        rag.get("pipe_from__first_step", rag.get("pipe_from"))
                                    ][ctx_id],
                                },
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                },
                            ],
                        }
                    )

                    # This happens when we keep the full history for circular
                    if isinstance(generated_label, list | tuple):
                        for gl in generated_label:
                            _res.append({"role": "assistant", "content": _get_conv(gl, prompt)})
                            _res.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": batched_contexts[ctx_id]}
                                    ],
                                }
                            )
                        _res = _res[:-1]  # remove last user prompt

                    else:
                        _res.append(
                            {"role": "assistant", "content": _get_conv(generated_label, prompt)}
                        )

                    _rag_data.append(_res)

            elif format == "all-image-answer-pairs":
                _rag_data = []

                for relabel_idx in range(len(text)):
                    _res = []
                    content = []

                    # Add pre-context prompt if specified
                    # If there's only one element, i.e., the one to be re-labelled, use another
                    # prompt that does not mention multiple images
                    if len(_data) - 1 == 0 and rag.get("context_prompt_no_samples") is not None:
                        content.append({"type": "text", "text": rag["context_prompt_no_samples"]})

                    # Standard case
                    elif rag.get("context_prompt") is not None:
                        content.append({"type": "text", "text": rag["context_prompt"]})

                    for in_context_idx, (_image, generated_label) in enumerate(
                        zip(_data, text, strict=True)
                    ):
                        # The image being relabeled will go last
                        if in_context_idx == relabel_idx:
                            continue

                        _, base64_string = self._prepare_visual_for_context(_image)

                        # Add the image
                        content.append(
                            {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}
                        )

                        # This happens when we keep the full history for circular
                        if isinstance(generated_label, list | tuple):
                            for gl in generated_label:
                                content.extend(_get_conv(gl, prompt))
                                content.append(
                                    {
                                        "type": "text",
                                        "text": batched_contexts[ctx_id],
                                    }
                                )

                            content = content[:-1]

                        else:
                            content.extend(_get_conv(generated_label, prompt))

                    # Add a post-context prompt if specified
                    if (
                        len(_data) - 1 == 0
                        and rag.get("end_context_prompt_no_samples") is not None
                    ):
                        content.append(
                            {"type": "text", "text": rag["end_context_prompt_no_samples"]}
                        )
                    elif rag.get("end_context_prompt") is not None:
                        content.append({"type": "text", "text": rag["end_context_prompt"]})

                    # Now add the image to be re-labelled
                    _image = _data[relabel_idx]
                    generated_label = text[relabel_idx]
                    _, base64_string = self._prepare_visual_for_context(_image)
                    content.append(
                        {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}
                    )

                    # Put the model's output from previous step
                    if rag.get("keep_label_for_hot_image", False):
                        if isinstance(generated_label, list | tuple):
                            for gl in generated_label:
                                content.extend(_get_conv(gl, rag.get("prompt_for_sample", prompt)))
                                content.append({"type": "text", "text": batched_contexts[ctx_id]})

                            content = content[:-1]

                        else:
                            content.extend(
                                _get_conv(generated_label, rag.get("prompt_for_sample", prompt))
                            )

                    # Add the conversation to the results
                    _res.append({"role": "user", "content": content})

                    _rag_data.append(_res)

            else:
                raise NotImplementedError(f"Format {format} not implemented for RAG `pipe_from`")

            retrieved_data.append(_rag_data)

        return retrieved_data

    def generate_until_multi_round(self, requests: list[TaskInstance]) -> list[str]:
        """Generate using multiple rounds of interaction.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, until). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        res = []
        last_layer_topk_by_round: dict[int, list[list[dict[str, float | str]] | None] | None] = {}

        gen_kwargs = requests[0].args[1]
        start_req = gen_kwargs.get("start_req", 0)
        end_req = gen_kwargs.get("end_req", len(requests))
        original_len = len(requests)
        requests = copy.deepcopy(requests[start_req:end_req])

        def _collate(x: tuple[str, ...]) -> tuple[int, str]:
            """Group and sort requests by context length for efficient batching.

            The negative sign on len(tokens) sorts in descending order, which provides several
                advantages:
                - Time estimates will be overestimates rather than underestimates, which is more
                    useful for planning;
                - The first item in a batch determines the padded context length, simplifying
                    batching logic;
                - Makes automatic adaptive batches much easier to implement;
                - Any out-of-memory errors occur immediately rather than near the end.

            Args:
            ----
                x: A tuple containing the context string and other arguments

            """
            tokens = self.tokenizer.encode(x[0])
            return -len(tokens), x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        global_rag_messages_by_round = {}
        global_answers_by_round = {}
        global_data_by_round = {}

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk_idx, chunk in enumerate(chunks):
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_to_text,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk, strict=True)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [
                batched_doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in batched_doc_id
            ]
            # Keep one visual payload per sample; do not flatten multi-image contexts.

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            if gen_kwargs.get("resize_input_image", None) is not None:
                for visual in batched_visuals:
                    for image in _iter_images(visual):
                        # Scale down, keeping aspect ratio, if the image is larger
                        if max(image.size) > gen_kwargs["resize_input_image"]:
                            image.thumbnail(
                                (gen_kwargs["resize_input_image"], gen_kwargs["resize_input_image"]),
                                Image.LANCZOS,
                            )

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        "Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            # Setup RAG for multi-round
            rag_all_steps = (gen_kwargs or {}).get("rag") or {}
            self._setup_rag(gen_kwargs, multi_step=True)

            # Setup mapping from round idx to actual step idx
            round_idx_to_actual_step = {}
            round_idx = 0
            random_sample_filenames_by_ctx: list[list[str]] = [
                [] for _ in range(len(batched_contexts))
            ]
            random_sample_targets_by_ctx: list[list[str]] = [
                [] for _ in range(len(batched_contexts))
            ]
            for step_idx, rag_config in rag_all_steps.items():
                if rag_config.get("circular", False):
                    for _ in range(rag_config.get("max_iters", 1)):
                        round_idx_to_actual_step[round_idx] = step_idx
                        round_idx += 1
                else:
                    round_idx_to_actual_step[round_idx] = step_idx
                    round_idx += 1

            # This happens when no RAG is used, so we simply map rounds to steps 1:1
            if rag_all_steps == {}:
                # Get the prompts for each step
                round_idx = len(
                    batched_doc_to_text[0]
                    .keywords.get("model_specific_kwargs", {})
                    .get("default", {})
                    .get("prompts", [])
                )
                round_idx_to_actual_step = {k: k for k in range(round_idx)}

            _max_round = round_idx

            # Store answers (generated text), conversation histories
            prompts_by_round = {}
            batched_round_results, batched_round_info = [], []

            # Store answers by round
            answers_by_round = {}
            data_by_round = {}

            round_idx = 0

            # Iterate rounds
            while True:
                last_round_info = None

                # Get RAG config and set it up for this round
                rag = rag_all_steps.get(round_idx_to_actual_step.get(round_idx, _max_round), {})
                rag_enabled = rag.get("enabled", False)
                rag["rag_enabled"] = rag_enabled
                rag["original_pipe_from"] = rag.get("pipe_from")
                if rag_enabled:
                    rag["doc_to_target"] = configurable_task.doc_to_target
                    rag["doc_to_visual"] = configurable_task.doc_to_visual
                    rag["test_docs"] = configurable_task.test_docs

                if rag.get("circular", False):
                    if rag.get("pipe_from") == -1:
                        rag["pipe_from__first_step"] = (
                            round_idx_to_actual_step.get(round_idx, _max_round) - 1
                        )
                    else:
                        rag["pipe_from__first_step"] = rag.get("original_pipe_from")

                if rag.get("pipe_from") == -1:
                    rag["pipe_from"] = round_idx - 1  # previous round

                # Rounds after the first one
                if round_idx != 0:
                    results = []
                    for ids_idx, doc_id in enumerate(batched_doc_id):
                        previous_round_results = [
                            round_results[ids_idx] for round_results in batched_round_results
                        ]
                        if len(batched_round_info) > 0:
                            last_round_info = batched_round_info[-1][ids_idx]

                        result = batched_doc_to_text[0](
                            self.task_dict[task][split][doc_id],
                            round_idx=round_idx_to_actual_step.get(round_idx, _max_round),
                            previous_round_results=previous_round_results,
                            last_round_info=last_round_info,
                        )
                        results.append(result)

                    (
                        _,  # batched_visuals, but we keep them for RAG.
                        # Just make sure to not put them in the prompt again
                        batched_contexts,
                        batched_terminal_signal,
                        batched_round_results,
                        last_round_info,
                    ) = list(zip(*results, strict=True))

                    batched_round_results = list(zip(*batched_round_results, strict=True))
                    if batched_terminal_signal[0]:  # terminal signal from doc_to_text function
                        break

                if isinstance(batched_contexts, tuple):
                    batched_contexts = list(batched_contexts)

                prompts_by_round[round_idx] = batched_contexts

                for i in range(len(batched_contexts)):
                    if "<image>" in batched_contexts[i]:
                        batched_contexts[i] = batched_contexts[i].replace("<image>", "")

                # ================= #
                # Data preparation  #
                # ================= #

                rag_messages = {}
                if rag_enabled:
                    passthrough_answers = None

                    # Get results from a previous step in the multi-round process
                    if rag.get("pipe_from") is not None:
                        # Using cached results from previous rounds
                        if (
                            rag.get("cache_and_reuse", False)
                            and round_idx in global_rag_messages_by_round
                        ):
                            rag_messages = global_rag_messages_by_round[round_idx]

                        # Passing through previous answers directly
                        elif rag.get("passthrough", False):
                            passthrough_answers = []
                            for ctx_id in range(len(batched_contexts)):
                                indices_of_circular = [
                                    k
                                    for k, v in round_idx_to_actual_step.items()
                                    if v == round_idx_to_actual_step.get(rag.get("pipe_from"))
                                    and k < round_idx
                                ]

                                indices_of_circular = sorted(list(set(indices_of_circular)))
                                text = []
                                for idx in indices_of_circular:
                                    _t = answers_by_round[idx]
                                    if isinstance(_t[0], list):
                                        _t = _t[ctx_id]
                                    text.append(_t)
                                text = list(zip(*text, strict=True))

                                if len(rag.get("concatenate_labels", "")) > 0:
                                    text = [
                                        rag.get("concatenate_labels", "").join(
                                            [t_item.strip() for t_item in t]
                                        )
                                        for t in text
                                    ]
                                else:
                                    text = [t[-1] for t in text]

                                # Get the label corresponding to the input image
                                text = text[rag.get("grab_outputs_from", -1)]
                                passthrough_answers.append(text)

                        # Actually load the data from a previous round
                        else:
                            retrieved_data = self._pipe_from(
                                rag,
                                batched_contexts,
                                round_idx,
                                round_idx_to_actual_step,
                                answers_by_round,
                                data_by_round,
                                prompts_by_round,
                            )

                            rag_messages = {}
                            for k, v in enumerate(retrieved_data):
                                rag_messages[k] = v

                            # Caching data for future rounds
                            if rag.get("cache_and_reuse", False):
                                global_rag_messages_by_round[round_idx] = rag_messages

                    # Standard RAG retrieval
                    else:
                        # Loading results from cache
                        if (
                            rag.get("cache_and_reuse", False)
                            and round_idx in global_rag_messages_by_round
                        ):
                            rag_messages = copy.deepcopy(global_rag_messages_by_round[round_idx])

                        # Performing actual retrieval
                        else:
                            rag_messages, rag_retrieved_filenames, rag_retrieved_targets = self._batch_rag(
                                batched_doc_id,
                                batched_contexts,
                                batched_visuals,
                                {"rag": rag},
                                task,
                                split,
                                round_idx,
                            )

                            if rag.get("search_modality") == "random" and rag.get("pipe_from") is None:
                                random_sample_filenames_by_ctx = [
                                    rag_retrieved_filenames.get(i, rag_retrieved_filenames.get(0, []))
                                    for i in range(len(batched_contexts))
                                ]
                                random_sample_targets_by_ctx = [
                                    rag_retrieved_targets.get(i, rag_retrieved_targets.get(0, []))
                                    for i in range(len(batched_contexts))
                                ]

                            if rag.get("cache_and_reuse", False):
                                global_rag_messages_by_round[round_idx] = copy.deepcopy(
                                    rag_messages
                                )

                        # Put the input images in the messages
                        # (which are a list of `PIL.Image`s in this case)
                        if rag.get("include_input", False):
                            for visual in batched_visuals:
                                image = _get_primary_image(visual)
                                if image is not None:
                                    rag_messages[0].append(image)

                        if rag.get("include_input_separately", False):
                            for k in rag_messages:
                                image = _get_primary_image(
                                    batched_visuals[k] if k < len(batched_visuals) else None
                                )
                                if image is not None:
                                    rag_messages[k].append(image)

                # ================= #
                # Answer generation #
                # ================= #

                # When there is RAG with separate queries for each retrieved document
                if rag_enabled and rag.get("separate_queries", False):
                    # Load answers from cache directly
                    # If the input is included, we cannot use cached answers directly -- it's
                    # handled in the `else` branch below
                    if (
                        rag.get("cache_and_reuse", False)
                        and round_idx in global_answers_by_round
                        and not rag.get("include_input", False)
                    ):
                        answers_by_round[round_idx] = global_answers_by_round[round_idx]
                        data_by_round[round_idx] = global_data_by_round[round_idx]
                        batched_round_results.append(answers_by_round[round_idx])

                    # In this case, each retrieved document is processed separately
                    # and we do not process the main query
                    else:
                        rag["rag_enabled"] = False  # Disable RAG for the side-query processing
                        answers_by_round[round_idx] = []
                        data_by_round[round_idx] = []

                        # If we are here, it means that we might find the answers in the cache,
                        # but we still have to process the new inputs (the "hot" images)
                        cached_answers, cached_data = None, None
                        if (
                            rag.get("cache_and_reuse", False)
                            and round_idx in global_answers_by_round
                        ):
                            cached_answers = copy.deepcopy(global_answers_by_round[round_idx])

                            # Since `cached_answers` contains the same answers repeated
                            # for each item in the batch, we can just keep the first one (`[0]`)
                            cached_answers = cached_answers[0][: rag.get("num_samples")]

                            # Same for `cached_data`
                            cached_data = global_data_by_round[round_idx][0]

                            # Remove the cached items from the current
                            # rag messages, so that we only process the new ones
                            rag_messages[0] = rag_messages[0][rag.get("num_samples") :]

                        # Process each item in the batch separately
                        for i, _visuals in enumerate(rag_messages.values()):
                            _history = None
                            if rag.get("format", "") in (
                                "image-answer-pairs",
                                "all-image-answer-pairs",
                            ):
                                _history = _visuals

                            # Construct the actual conversation for the model
                            _contexts = [batched_contexts[i]] * len(_visuals)
                            messages, images_per_request = self._make_history(
                                gen_kwargs,
                                _contexts,
                                _visuals,
                                rag,
                                rag_messages,
                                history=_history,
                                append_history=True,
                                step_idx=round_idx,
                            )

                            # Using sub-batches to handle requests for this batch item
                            if rag.get("gen_batch_size", None) is not None:
                                # Process in smaller batches to avoid OOM
                                answers = []
                                perplexities = []
                                for j in range(0, len(messages), rag["gen_batch_size"]):
                                    _inputs, _generation_output = self._generate(
                                        messages[j : j + rag["gen_batch_size"]],
                                        gen_kwargs,
                                        rag,
                                    )
                                    answers.append(self._decode(_inputs, _generation_output))
                                    perplexities.extend(
                                        self._loglikelihood(_inputs, _generation_output)[
                                            "perplexity"
                                        ]
                                    )

                                    del _inputs, _generation_output
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

                                answers = _flatten_list(answers)

                            # Process all at once
                            else:
                                inputs, generation_output = self._generate(
                                    messages, gen_kwargs, rag
                                )
                                answers = self._decode(inputs, generation_output)

                                del inputs, generation_output
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                            # Post-process the answers and store them
                            for i, answer in enumerate(answers):
                                for term in until:
                                    if len(term) > 0:
                                        answer = answer.split(term)[0]

                                if "," in answer and rag.get("clean_output_labels", False):
                                    answer = answer.split(",")

                                    # Keep only the last non-empty label
                                    for test_idx in reversed(range(len(answer))):
                                        temp_answer = answer[test_idx].strip().replace('"', "")
                                        if len(temp_answer) > 0:
                                            answer = temp_answer
                                            break

                                    # Fallback in case all labels were empty after cleaning
                                    if isinstance(answer, list):
                                        answer = answer[0]

                                answers[i] = answer

                            # If we are using cached answers for the fixed items, we have
                            # to store them as if we generated them now. Since fixed images
                            # are the beginning of the batch, we prepend them to the ones
                            # we've just generated
                            if cached_answers is not None:
                                answers = cached_answers + answers
                            if cached_data is not None:
                                _visuals = cached_data + _visuals

                            answers_by_round[round_idx].append(answers)
                            data_by_round[round_idx].append(_visuals)

                        # If caching is enabled and we include input, prepare
                        # the answers and data accordingly
                        if rag.get("cache_and_reuse", False) and rag.get("include_input", False):
                            # Here, `len(rag_messages)` should be 1
                            answers_by_round[round_idx] = answers_by_round[round_idx] * len(
                                batched_contexts
                            )
                            data_by_round[round_idx] = data_by_round[round_idx] * len(
                                batched_contexts
                            )

                            # Now, for each list in the two lists, retain only
                            # the first N items and then the i-th one
                            retain_first = rag.get("num_samples")
                            for i in range(len(answers_by_round[round_idx])):
                                item_to_retain = retain_first + i
                                answers_by_round[round_idx][i] = (
                                    answers_by_round[round_idx][i][:retain_first]
                                    + answers_by_round[round_idx][i][
                                        item_to_retain : item_to_retain + 1
                                    ]
                                )
                                data_by_round[round_idx][i] = (
                                    data_by_round[round_idx][i][:retain_first]
                                    + data_by_round[round_idx][i][
                                        item_to_retain : item_to_retain + 1
                                    ]
                                )

                        # When caching and reusing results, reuse the first
                        # result for all items in the batch
                        if rag.get("cache_and_reuse", False):
                            # When the input is not included, i.e., when all items are
                            # *actually* the same, we can just reuse the cached answers
                            # faking answers for all items in the batch
                            if not rag.get("include_input", False):
                                for _ in range(1, len(batched_contexts)):
                                    answers_by_round[round_idx].append(
                                        answers_by_round[round_idx][0]
                                    )
                                    data_by_round[round_idx].append(data_by_round[round_idx][0])

                            if round_idx not in global_answers_by_round:
                                global_answers_by_round[round_idx] = answers_by_round[round_idx]
                                global_data_by_round[round_idx] = data_by_round[round_idx]

                        batched_round_results.append(answers_by_round[round_idx])

                        rag["rag_enabled"] = True  # Re-enable RAG for compatibility

                # Standard generation process
                else:
                    gen_metrics = None
                    context_tokens_count = None

                    if rag_enabled and rag.get("passthrough", False):
                        # Special case where circular has been used to generate labels: we just
                        # pass the result through here
                        if passthrough_answers is None:
                            raise ValueError(
                                "`passthrough_answers` cannot be None "
                                "when `rag['passthrough']` is `True`"
                            )
                        answers = passthrough_answers
                        answers_by_round[round_idx] = answers
                        inputs = None
                        generation_output = None
                        messages = [[]] * len(answers)

                    else:
                        # Standard case: we process the main query (i.e., the context and
                        # provided visuals) with optional retrieved documents if RAG is enabled
                        messages, images_per_request = self._make_history(
                            gen_kwargs,
                            batched_contexts,
                            batched_visuals,
                            copy.deepcopy(rag),
                            copy.deepcopy(rag_messages),
                            history=last_round_info,
                            step_idx=round_idx,
                        )

                        inputs, generation_output = self._generate(messages, gen_kwargs, rag)
                        if gen_kwargs.get("print_last_layer_probs", False):
                            last_layer_topk_by_round[round_idx] = self._print_last_layer_probs(
                                generation_output,
                                topk=gen_kwargs.get("last_layer_probs_topk", 5),
                                max_samples=gen_kwargs.get("last_layer_probs_max_samples", 3),
                                task=task,
                                split=split,
                                doc_ids=list(batched_doc_id),
                            )
                        answers = self._decode(inputs, generation_output)
                        gen_metrics = self._loglikelihood(inputs, generation_output)

                        context_tokens_count = [
                            inputs["attention_mask"][idx].sum()
                            for idx in range(len(batched_contexts))
                        ]

                        del inputs, generation_output
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    answers_by_round[round_idx] = answers

                    # Get clean answers and store them in the conversation history
                    for i, answer in enumerate(answers):
                        for term in until:
                            if len(term) > 0:
                                answer = answer.split(term)[0]

                        # Remove duplicates
                        if rag.get("clean_answer", False):
                            seen = set()
                            seen_add = seen.add

                            answer = answer.strip()
                            answer = answer.split(",")
                            answer = ", ".join(
                                [
                                    a.strip().replace('"', "")
                                    for a in answer
                                    if len(a.strip()) > 0
                                    if not (a in seen or seen_add(a))
                                ]
                            )

                        answers[i] = answer
                        messages[i].append(
                            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
                        )

                    batched_round_results.append(answers)
                    batched_round_info.append(messages)

                    if rag.get("store_memory", False):
                        self._retriever[0].store_memory(batched_visuals, answers)

                reset_cache = rag.get("reset_cache", False)
                # Check if it's a bool or an int
                # If it's an int, we reset the cache every `reset_cache` batches
                # This is like having a larger batch size
                if not isinstance(reset_cache, bool):
                    reset_cache = (chunk_idx + 1) % reset_cache == 0
                if reset_cache:
                    global_rag_messages_by_round = {}
                    global_answers_by_round = {}
                    global_data_by_round = {}

                round_idx += 1

                # Restore original `pipe_from` to correctly support circular contexts
                rag["pipe_from"] = rag.get("original_pipe_from")

                # Cleanup RAG keys
                rag.pop("doc_to_target", None)
                rag.pop("doc_to_visual", None)
                rag.pop("test_docs", None)

            answers = list(zip(*batched_round_results, strict=True))
            for idx, (ans, context) in enumerate(zip(answers, batched_contexts, strict=True)):
                last_layer_topk_probs = None
                if last_layer_topk_by_round:
                    round_count = len(batched_round_results)
                    per_round = []
                    for round_idx in range(round_count):
                        round_data = last_layer_topk_by_round.get(round_idx)
                        per_round.append(round_data[idx] if round_data is not None else None)
                    last_layer_topk_probs = per_round

                _ans = TaskSingleOutput(
                    answer=ans,
                    last_layer_topk_probs=last_layer_topk_probs,
                    context=context,
                    context_tokens_count=context_tokens_count[idx]
                    if context_tokens_count is not None
                    else None,
                    num_images=images_per_request[idx] if len(images_per_request) > idx else None,
                    loglikelihood=gen_metrics["loglikelihood"][idx].item()
                    if gen_metrics is not None
                    else None,
                    perplexity=gen_metrics["perplexity"][idx].item()
                    if gen_metrics is not None
                    else None,
                    rag_sample_filenames=random_sample_filenames_by_ctx[idx],
                    rag_sample_targets=random_sample_targets_by_ctx[idx],
                )
                res.append(_ans)
                self.cache_hook.add_partial(
                    "generate_until_multi_round", (context, gen_kwargs), ans
                )
                pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        # Adding back dummy results for skipped requests
        # Doing this after reordering because the `reordered` object only
        # knows about the actual requests being processed
        if start_req > 0:
            res = [TaskSingleOutput(answer=[""], is_dummy=True) for _ in range(start_req)] + res
        if end_req < original_len:
            res = res + [
                TaskSingleOutput(answer=[""], is_dummy=True) for _ in range(original_len - end_req)
            ]

        pbar.close()
        return res


@register_model("qwen2-vl-7b")
def qwen2_vl_7b(**model_kwargs) -> Model:
    """Load the Qwen2VL model with 7B params."""
    model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2-vl-2b")
def qwen2_vl_2b(**model_kwargs) -> Model:
    """Load the Qwen2VL model with 2B params."""
    model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2.5-vl-7b")
def qwen25_vl_7b(**model_kwargs) -> Model:
    """Load the Qwen2.5VL model with 7B params."""
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2.5-vl-3b")
def qwen25_vl_3b(**model_kwargs) -> Model:
    """Load the Qwen2.5VL model with 3B params."""
    model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen3-vl-8b")
def qwen3_vl_8b(**model_kwargs) -> Model:
    """Load the Qwen3VL model with 8B params."""
    model_name_or_path = "Qwen/Qwen3-VL-8B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model

