import copy
import json
import os
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from packaging import version
from PIL import Image
from transformers.utils import ModelOutput

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

__all__ = ["LLaVAOnevision"]

log = utils.get_logger(__name__, rank_zero_only=True)


def _is_json(string: str) -> bool:
    """Check if a string is a valid JSON.

    Args:
    ----
        string (str): The string to check.

    """
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False


class LLaVAOnevision(Model):
    """LLaVA-Onevision model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "lmms-lab/llava-onevision-qwen2-7b-ov".
        model_name (str, optional): Name of the model. Defaults to None.
        attn_implementation (str, optional): The attention implementation to use. Defaults to
            "sdpa" for torch>=2.1.2, "eager" otherwise.
        conv_template (str, optional): Template for formatting conversations. Defaults to
            "qwen_1_5".
        use_flash_attn_2 (bool): Whether to use flash attention 2. Defaults to False.
        use_cache (bool, optional): Whether to use KV cache during generation. Defaults to True.
        customized_config (str, optional): Path to custom configuration JSON file. Defaults to
            None.
        mm_spatial_pool_stride (int, optional): Stride for spatial pooling. Defaults to 2.
        mm_spatial_pool_mode (str, optional): Mode for spatial pooling. Defaults to "bilinear".
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        model_name: str | None = None,
        attn_implementation: str | None = (
            "flash_attention_2"
            if utils.package_available("flash_attn")
            else (
                "sdpa" if version.parse(torch.__version__) >= version.parse("2.1.2") else "eager"
            )
        ),
        use_flash_attn_2: bool = utils.package_available("flash_attn"),
        conv_template: str | None = "qwen_1_5",
        use_cache: bool | None = True,
        customized_config: str | None = None,
        mm_spatial_pool_stride: int | None = 2,
        mm_spatial_pool_mode: str | None = "bilinear",
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._model_name = (
            model_name if model_name is not None else get_model_name_from_path(model_name_or_path)
        )
        self._attn_implementation = attn_implementation
        self._use_flash_attn_2 = use_flash_attn_2
        self._conv_template = conv_template
        self._use_cache = use_cache
        self._customized_config = customized_config
        self._mm_spatial_pool_stride = mm_spatial_pool_stride
        self._mm_spatial_pool_mode = mm_spatial_pool_mode

        self._max_length = None

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU", "DEEPSPEED"],
            **kwargs,
        )

        torch.backends.cuda.matmul.allow_tf32 = True

    def _pad_sequence(
        self, input_ids: list[torch.Tensor], batch_first: bool, padding_value: int
    ) -> torch.Tensor:
        """Pad a list of variable length tensors.

        Custom padder that handles left and right padding based on tokenizer setting.
        Left-padding tensors are flipped before and after the padding operation.

        Args:
        ----
            input_ids (list[torch.Tensor]): List of input tensors to pad.
            batch_first (bool): Whether output should be batch first (B, T) or sequence first
                (T, B).
            padding_value (int): Value used for padding.

        """
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def _tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """Encode a string into tokens using the model's tokenizer.

        Args:
        ----
            string (str): The input string to encode
            left_truncate_len (int, optional): If provided, truncate the encoded tokens from the
                left to this length. Defaults to None.
            add_special_tokens (bool, optional): Whether to add special tokens during encoding. If
                None, defaults to False. Defaults to None.

        """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # Left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _tok_decode(self, tokens: int | list[int]) -> str:
        """Decode token(s) to text string using the model's tokenizer.

        Args:
        ----
            tokens: A single token ID or list of token IDs to decode.

        """
        try:
            return self.tokenizer.decode(tokens)
        except ValueError:
            return self.tokenizer.decode([tokens])

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "multimodal": True,
            "use_flash_attention_2": self._use_flash_attn_2,
            "torch_dtype": str(self.dtype).split(".")[1],
            "device_map": self.device_map,
        }

        if self._customized_config is not None:
            model_kwargs["customized_config"] = self._customized_config
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation
        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self._mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self._mm_spatial_pool_mode

        model_kwargs["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multi-modal argument
            tokenizer, model, processor, max_length = load_pretrained_model(
                self._model_name_or_path,
                None,
                self._model_name,
                **model_kwargs,
            )
            model = torch.compile(model, mode="max-autotune", fullgraph=True)

            self._model = model
            self._processor = processor
            self._tokenizer = tokenizer
            self._max_length = max_length
        except TypeError:
            # Older versions of LLaVA don't have multi-modal argument
            model_kwargs.pop("multimodal", None)
            tokenizer, model, processor, max_length = load_pretrained_model(
                self._model_name_or_path,
                None,
                self._model_name,
                **model_kwargs,
            )
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            self._model = model
            self._processor = processor
            self._tokenizer = tokenizer
            self._max_length = max_length

        # To support batching
        model.config.tokenizer_padding_side = "left"

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
        res = []

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        origin_image_aspect_ratio = getattr(self.config, "image_aspect_ratio", None)

        reg_args = [reg.args for reg in requests]
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in reg_args:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            wrong_aspect_ratio = self.config.image_aspect_ratio != origin_image_aspect_ratio
            if origin_image_aspect_ratio is not None and wrong_aspect_ratio:
                self.config.image_aspect_ratio = origin_image_aspect_ratio
                log.info("Resetting image aspect ratio to %s", origin_image_aspect_ratio)

            if visual is None or visual == []:
                visual = None
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self.config.__dict__:
                    self.config.image_aspect_ratio = "pad"
                    log.info(
                        "In Multi-Image setting, image aspect ratio: %s",
                        self.config.image_aspect_ratio,
                    )

                image_tensor = process_images(visual, self.processor, self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [
                        _image.to(dtype=self.dtype, device=self.device) for _image in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

            is_image_defined = image_tensor is not None and len(image_tensor) != 0
            if is_image_defined and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self._conv_template:
                conv = copy.deepcopy(conv_templates[self._conv_template])
            else:
                conv = conv_templates[self._conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = (
                tokenizer_image_token(
                    full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            if visual is None:
                raise ValueError("visual cannot be None.")

            kwargs = {}
            kwargs["image_sizes"] = (
                [[v.size[0], v.size[1]] for v in visual]
                if isinstance(visual, list)
                else [[visual.size[0], visual.size[1]]]
            )

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=full_input_ids,
                    labels=labels,
                    images=image_tensor,
                    use_cache=True,
                    **kwargs,
                )

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            continuation_tokens = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            max_equal = (greedy_tokens == continuation_tokens).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def _loglikelihood(
        self, input_ids: dict, generation_output: ModelOutput
    ) -> dict[str, torch.Tensor | list]:
        """Compute log-likelihood and related metrics for generated sequences.

        Args:
        ----
            input_ids: The input tensors used for generation.
            generation_output: The output from the model's generate method.

        """
        cont = generation_output.sequences
        scores = generation_output.scores

        transition_scores = self.model.compute_transition_scores(
            cont,
            scores,
            normalize_logits=True,  # ensure proper log probabilities (softmax is applied)
        )

        # Extract generated tokens
        # For LLaVa OV, the prompt is not included in `cont`, so no need
        # to remove the length of `input_ids` from each response
        generated_tokens = cont

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
            eos_positions = (
                (gen_seq == self.tokenizer.eos_token_id) | (gen_seq == self.tokenizer.pad_token_id)
            ).nonzero(as_tuple=True)[0]
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
    ) -> None | list[dict[str, Any]]:
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
            if resize_image_size is not None and max(image.size) > resize_image_size:
                image.thumbnail((resize_image_size, resize_image_size), Image.LANCZOS)

            payload = {
                "type": "image",
                "image": image,
            }

            return payload

        def result_callback(
            rag_data: list[dict[str, Any]], images: list | None
        ) -> dict[str, list]:
            # Keep only the text elements and add a placeholder for images if any
            _rag_data = []
            for elem in rag_data:
                if elem.get("type") == "text":
                    _rag_data.append(elem.get("text"))
                elif elem.get("type") == "image" and images is not None:
                    _rag_data.append("<images>")

            return {
                "rag_data": _rag_data,
                "images": [x.get("image") for x in images] if images is not None else [],
            }

        prepare = partial(
            _prepare,
            resize_image_size=(gen_kwargs or {}).get("rag", {}).get("resize_images", None),
        )

        retriever = self._retriever
        if step_idx is not None:
            retriever = retriever[step_idx]
        result = retriever.retrieve_and_prepare(
            gen_kwargs,
            images,
            prepare=prepare,  # always using custom `prepare` function for LLaVa OV
            result_callback=result_callback,
            doc_ids=doc_ids,
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        result = list(cast(list[dict[str, list]], result))

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
    ) -> dict:
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
            if visual is not None and len(visual) == 1 and isinstance(visual[0], Image.Image):
                visual = visual[0].convert("RGB")
                batched_visuals_for_rag[i] = visual

                # When caching and reusing, only retrieve for the first image
                # Typically, this is used with random sampling, so we there's no
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

        return rag_messages

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
    ) -> tuple[list, list, list, list]:
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

        origin_image_aspect_ratio = getattr(self.config, "image_aspect_ratio", None)

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

        convs = []
        question_input = []
        _batched_visuals = []
        image_tensor = None

        for i, (visual, context) in enumerate(zip(batched_visuals, batched_contexts, strict=True)):  # noqa: E501
            _batched_visual_to_add = visual
            wrong_aspect_ratio = self.config.image_aspect_ratio != origin_image_aspect_ratio
            if origin_image_aspect_ratio is not None and wrong_aspect_ratio:
                self.config.image_aspect_ratio = origin_image_aspect_ratio
                log.info("Resetting image aspect ratio to %s", origin_image_aspect_ratio)

            if visual is None or visual == []:  # For text-only tasks.
                visual = None
                placeholder_count = 0
                image_tensor = None
            else:
                # Make it a list for uniform processing
                if isinstance(visual, Image.Image):
                    visual = [visual]

                # For multi image case, we treat per image aspect ratio as "pad" by default.
                if len(visual) > 1 or "image_aspect_ratio" not in self.config.__dict__:
                    self.config.image_aspect_ratio = getattr(
                        gen_kwargs, "image_aspect_ratio", "pad"
                    )
                    log.info(
                        "In Multi-Image setting, image aspect ratio: %s",
                        self.config.image_aspect_ratio,
                    )

                # Here in `_make_history`, `image_tensor` is only used for checking if
                # the image is actually present or not. The actual `image_tensor` used
                # for generation is created in `_generate`, unless batch size = 1, in
                # which case, the `image_tensor` created here is returned and used
                # directly for generation.
                image_tensor = process_images(visual, self.processor, self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [
                        _image.to(dtype=self.dtype, device=self.device) for _image in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

                placeholder_count = len(visual) if isinstance(visual, list) else 1

            rag_message = None
            if rag_enabled:
                if i in rag_messages:
                    rag_message = rag_messages[i]
                else:
                    rag_message = self._retrieve(gen_kwargs, visual, step_idx=step_idx)[0]
                # Get the actual message and the images
                rag_message, rag_images = rag_message["rag_data"], rag_message["images"]

            is_image_defined = image_tensor is not None and len(image_tensor) != 0
            # Disable image after first step when RAG is not enabled
            if not rag_enabled and step_idx is not None and step_idx > 0:
                is_image_defined = False
            if (
                is_image_defined
                and DEFAULT_IMAGE_TOKEN not in context
                and (not rag_enabled or rag.get("include_image", True))
            ):
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)

                if include_target_classes and include_target_classes_position == "pre-query":
                    question = (
                        image_tokens
                        + "\n"
                        + target_classes_prompt.format(target_classes_str)
                        + "\n"
                        + context
                    )
                elif include_target_classes and include_target_classes_position == "end-of-ctx":
                    question = (
                        image_tokens
                        + "\n"
                        + context
                        + "\n"
                        + target_classes_prompt.format(target_classes_str)
                    )
                else:
                    question = image_tokens + "\n" + context
            else:
                image_tokens = ""

                if include_target_classes and include_target_classes_position == "pre-query":
                    question = target_classes_prompt.format(target_classes_str) + "\n" + context
                elif include_target_classes and include_target_classes_position == "end-of-ctx":
                    question = context + "\n" + target_classes_prompt.format(target_classes_str)
                else:
                    question = context

            # When RAG is enabled and there is retrieved content, we can erase the original
            # question, and reconstruct it later by putting the retrieved content in the
            # right place
            if rag_enabled and rag_message is not None:
                question = ""

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self._conv_template:
                conv = copy.deepcopy(conv_templates[self._conv_template])
            else:
                conv = conv_templates[self._conv_template].copy()

            if history is not None and history[i] is not None:
                if append_history:
                    if isinstance(history[i], list):
                        for msg in history[i]:
                            # Append messages one by one with user role
                            conv.append_message(conv.roles[0], msg)
                    elif isinstance(history[i], str):
                        # Append the message with user role
                        conv.append_message(conv.roles[0], history[i])
                    else:
                        conv.messages.extend(history[i].messages)
                else:
                    conv = copy.deepcopy(history[i])

            # RAG is enabled
            if rag_enabled and rag_message is not None:
                if include_target_classes and include_target_classes_position == "pre-query":
                    context = target_classes_prompt.format(target_classes_str) + "\n" + context
                elif include_target_classes and include_target_classes_position == "end-of-ctx":
                    context = context + target_classes_prompt.format(target_classes_str)

                rag_message = "\n".join(rag_message).replace(
                    "<images>", DEFAULT_IMAGE_TOKEN + "\n"
                )

                # Retrieved data goes at the beginning of the context
                if rag_position == "pre-sample":
                    if len(rag_images) > 0:
                        _batched_visual_to_add = rag_images + _batched_visual_to_add

                    question = "".join(
                        [
                            rag_message,
                            "\n",
                            image_tokens,
                            "\n",
                            context,
                        ]
                    )

                # Retrieved data goes after the image, before the query
                if rag_position == "post-sample":
                    if len(rag_images) > 0:
                        _batched_visual_to_add = _batched_visual_to_add + rag_images

                    question = "".join(
                        [
                            image_tokens,
                            "\n",
                            rag_message,
                            "\n",
                            context,
                        ]
                    )

                # Retrieved data goes at the end of the context
                if rag_position == "post-sample-and-query":
                    if len(rag_images) > 0:
                        _batched_visual_to_add = _batched_visual_to_add + rag_images

                    question = "".join(
                        [
                            image_tokens,
                            "\n",
                            context,
                            "\n",
                            rag_message,
                        ]
                    )

            # Store the updated visuals (might have added RAG images)
            _batched_visuals.append(_batched_visual_to_add)

            if _is_json(question):  # Conversational question input
                question = json.loads(question)
                for idx, item in enumerate(question):
                    role = conv.roles[idx % 2]
                    message = item["value"]
                    conv.append_message(role, message)

                if len(conv.messages) % 2 != 1:
                    raise ValueError("Number of messages must be odd.")

                conv.append_message(conv.roles[1], None)
                convs.append(conv)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)
            else:  # Only simple string for question
                if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0]:
                    # Last message is from user, append to it
                    conv.messages[-1][1] += "\n" + question
                else:
                    conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                convs.append(conv)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

        # Note: `image_tensor` here is only for the last instance in the batch,
        # but when batch size = 1 this is sufficient
        return question_input, _batched_visuals, image_tensor, convs

    def _generate(
        self,
        question_input: list,
        _batched_visuals: list,
        image_tensor: list,
        gen_kwargs: dict,
        rag: dict,
        use_given_image_tensor: bool = False,
    ) -> tuple[dict, Any]:
        """Generate model outputs for the given messages.

        Args:
        ----
            question_input (list): Conversation for each sample in the batch.
            _batched_visuals (list): List of visual inputs for each sample in the batch.
            image_tensor (list): List of preprocessed image tensors for each sample in the batch.
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            rag (dict): RAG configuration dictionary.
            use_given_image_tensor (bool): Whether to use the given image tensor directly
                for generation.

        """
        # Use the updated batched visuals (might have added RAG images)
        batched_visuals = _batched_visuals

        rag_enabled = rag["rag_enabled"]

        # Recreate the image tensor correctly, assuming there's only one visual per instance
        # Otherwise, one might just use batch size = 1
        if rag.get("force_recreate_image_tensor", False) or (
            (self.batch_size > 1 or (rag_enabled and len(batched_visuals[0]) > 1))
            and not use_given_image_tensor
        ):
            image_tensor = process_images(sum(batched_visuals, []), self.processor, self.config)
            image_tensors = [
                _image.to(dtype=self.dtype, device=self.device) for _image in image_tensor
            ]
        else:
            image_tensors = image_tensor

        # Pre-configure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for prompt in question_input
        ]
        pad_token_ids = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = self._pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_ids
        ).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        gen_kwargs["image_sizes"] = [
            visual.size for visual in sum(batched_visuals, []) if visual is not None
        ]

        # These steps are not in LLaVA's original code, but are necessary for generation
        # to work.
        if "image_aspect_ratio" in gen_kwargs:
            gen_kwargs.pop("image_aspect_ratio")

        # Assume each instance in the batch has an image
        # Solution from: https://github.com/LLaVA-VL/LLaVA-NeXT/issues/169#issuecomment-2357309833
        modalities = ["image"] * len(sum(batched_visuals, []))

        with torch.inference_mode():
            generation_output = self.model.generate(
                input_ids,
                modalities=modalities,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensors,
                use_cache=self._use_cache,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )

        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "modalities": modalities,
            "image_sizes": gen_kwargs["image_sizes"],
        }, generation_output

    def _decode(self, generation_output: ModelOutput) -> list[str]:
        """Decode the generated output into a list of strings.

        Args:
        ----
            generation_output (ModelOutput): The output from the model generation.

        """
        text_outputs = self.tokenizer.batch_decode(
            generation_output.sequences, skip_special_tokens=True
        )
        text_outputs = [response.strip() for response in text_outputs]

        return text_outputs

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
            tokens = self._tok_encode(x[0])
            return -len(tokens), x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # RAG setup (guard-style, avoids deep nesting later)
            rag_enabled = self._is_rag_enabled(gen_kwargs)
            rag = (gen_kwargs or {}).get("rag") or {}
            rag["rag_enabled"] = rag_enabled
            if rag_enabled:
                rag = (gen_kwargs or {}).get("rag") or {}
                rag["doc_to_target"] = configurable_task.doc_to_target
                rag["doc_to_visual"] = configurable_task.doc_to_visual
                rag["test_docs"] = configurable_task.test_docs
                self._setup_rag(gen_kwargs)

            # RAG optimization: pre-load all images and retrieve in batches
            rag_messages = {}
            if rag_enabled:
                rag_messages = self._batch_rag(
                    batched_doc_id, batched_contexts, batched_visuals, gen_kwargs, task, split
                )

            if gen_kwargs.get("resize_input_image", None) is not None:
                for sublist_idx in range(len(batched_visuals)):
                    for img_idx in range(len(batched_visuals[sublist_idx])):
                        if not isinstance(batched_visuals[sublist_idx][img_idx], Image.Image):
                            continue

                        # Scale down, keeping aspect ratio, if the image is larger
                        if (
                            max(batched_visuals[sublist_idx][img_idx].size)
                            > gen_kwargs["resize_input_image"]
                        ):
                            batched_visuals[sublist_idx][img_idx].thumbnail(
                                (
                                    gen_kwargs["resize_input_image"],
                                    gen_kwargs["resize_input_image"],
                                ),
                                Image.LANCZOS,
                            )

            question_input, _batched_visuals, image_tensor, convs = self._make_history(
                gen_kwargs,
                batched_contexts,
                batched_visuals,
                rag,
                rag_messages,
            )

            images_per_request = [len(x) for x in _batched_visuals]  # For stats

            gen_kwargs_rag, resize_input_image = None, None
            if "rag" in gen_kwargs:
                gen_kwargs_rag = gen_kwargs.pop("rag")
            if "resize_input_image" in gen_kwargs:
                resize_input_image = gen_kwargs.pop("resize_input_image")

            _inputs, generation_output = self._generate(
                question_input, _batched_visuals, image_tensor, gen_kwargs, rag
            )
            input_ids = _inputs["input_ids"]
            attention_masks = _inputs["attention_masks"]

            context_tokens_count = attention_masks.sum(dim=1)

            text_outputs = self._decode(generation_output)
            gen_metrics = self._loglikelihood(input_ids, generation_output)

            del _inputs, generation_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if gen_kwargs_rag is not None:
                gen_kwargs["rag"] = gen_kwargs_rag
            if resize_input_image is not None:
                gen_kwargs["resize_input_image"] = resize_input_image

            # Cleanup RAG keys
            rag.pop("doc_to_target", None)
            rag.pop("doc_to_visual", None)
            rag.pop("test_docs", None)

            for idx, (ans, context) in enumerate(zip(text_outputs, batched_contexts, strict=True)):
                _ans = TaskSingleOutput(
                    answer=ans,
                    context=context,
                    context_tokens_count=context_tokens_count[idx].item(),
                    num_images=images_per_request[idx],
                    loglikelihood=gen_metrics["loglikelihood"][idx].item(),
                    perplexity=gen_metrics["perplexity"][idx].item(),
                )
                res.append(_ans)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # Reorder this group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res

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

    def _pipe_from(
        self,
        rag: dict,
        batched_contexts: list,
        round_idx: int,
        round_idx_to_actual_step: dict,
        answers_by_round: list,
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

        def _get_conv(text: str, prompt: str) -> list[str]:
            """Construct a conversation message from text and a prompt.

            Args:
            ----
                text (str): The text to include in the message.
                prompt (str): The prompt template, with a `{}` placeholder for the text.

            """
            return [prompt.format(text)]

        format = rag.get("format", "list-captions")
        prompt = rag.get("prompt", "{}")
        prompts = prompt.split("<images>")

        # When caching and reusing the retrieved data, we only need to
        # process the first item in the batch, and then reuse the same
        # data for all items in the batch
        _iterator = (
            range(len(batched_contexts)) if not rag.get("cache_and_reuse", False) else range(1)
        )
        retrieved_data = []
        for ctx_id in _iterator:
            if rag.get("keep_full_history", False):
                if rag.get("circular", False):
                    indices_of_circular = [
                        k
                        for k, v in round_idx_to_actual_step.items()
                        if v == round_idx_to_actual_step.get(round_idx) and k < round_idx
                    ]
                    indices_of_circular = [rag.get("pipe_from__first_step")] + indices_of_circular

                else:
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
                        rag.get("concatenate_labels", "").join([t_item.strip() for t_item in t])
                        for t in text
                    ]

            # Not keeping full history
            else:
                text = answers_by_round[rag.get("pipe_from")]

                # In case text is a list of lists (e.g., multiple answers for each item
                # in the batch), select the right one
                if isinstance(text[0], list):
                    text = text[ctx_id]

            # Get the data to go with this text
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

                resize_image_size = rag.get("resize_images")
                if resize_image_size is not None:
                    for img_idx in range(len(_data)):
                        if not isinstance(_data[img_idx], Image.Image):
                            continue

                        # Scale down, keeping aspect ratio, if the image is larger
                        if max(_data[img_idx].size) > resize_image_size:
                            _data[img_idx].thumbnail(
                                (resize_image_size, resize_image_size), Image.LANCZOS
                            )

            if rag.get("exclude_rag_message") is not None:
                text.pop(rag.get("exclude_rag_message"))
                _data.pop(rag.get("exclude_rag_message"))

            if format == "list-captions":
                text = [f"{idx + 1}. {t.strip()}" for idx, t in enumerate(text)]
                text = "\n".join(text)
                _rag_data = {
                    "rag_data": _get_conv(text, prompt),
                    "images": _data,
                }

            elif format == "bullet-list-captions":
                text = [f"- {t.strip()}" for t in text]
                text = "\n".join(text)
                _rag_data = {
                    "rag_data": _get_conv(text, prompt),
                    "images": [],
                }

            elif format == "newlines":
                text = "\n".join([t.strip() for t in text])
                _rag_data = {
                    "rag_data": _get_conv(text, prompt),
                    "images": _data,
                }

            elif format == "csv":
                text = ", ".join([t.strip() for t in text])
                _rag_data = {
                    "rag_data": _get_conv(text, prompt),
                    "images": _data,
                }

            elif format == "interleaved":
                content = [prompts[0]]
                images = []
                for _image, generated_label in zip(_data, text, strict=True):
                    content.append("<image>")
                    images.append(_image)
                    content.append(prompts[-1].format(generated_label))

                if rag.get("end_context_prompt") is not None:
                    content.append(rag["end_context_prompt"])

                _rag_data = {
                    "rag_data": content,
                    "images": images,
                }

            elif format == "image-answer-pairs":
                raise NotImplementedError("Implementation has to be adapted from Qwen2-VL.")

            elif format == "all-image-answer-pairs":
                _rag_data = []

                for relabel_idx in range(len(text)):
                    content = []
                    images = []

                    # Add pre-context prompt if specified
                    # If there's only one element, i.e., the one to be re-labelled, use another
                    # prompt that does not mention multiple images
                    if len(_data) - 1 == 0 and rag.get("context_prompt_no_samples") is not None:
                        content.append(rag["context_prompt_no_samples"])

                    # Standard case
                    elif rag.get("context_prompt") is not None:
                        content.append(rag["context_prompt"])

                    for in_context_idx, (_image, generated_label) in enumerate(
                        zip(_data, text, strict=True)
                    ):
                        # The image being relabeled will go last
                        if in_context_idx == relabel_idx:
                            continue

                        # Add the image
                        content.append("<image>")
                        images.append(_image)

                        # This happens when we keep the full history for circular
                        if isinstance(generated_label, list | tuple):
                            for gl in generated_label:
                                content.extend(_get_conv(gl, prompt))
                                content.append(batched_contexts[ctx_id])

                            content = content[:-1]

                        else:
                            content.extend(_get_conv(generated_label, prompt))

                    # Add a post-context prompt if specified
                    if (
                        len(_data) - 1 == 0
                        and rag.get("end_context_prompt_no_samples") is not None
                    ):
                        content.append(rag["end_context_prompt_no_samples"])
                    elif rag.get("end_context_prompt") is not None:
                        content.append(rag["end_context_prompt"])

                    # Now add the image to be re-labelled
                    _image = _data[relabel_idx]
                    generated_label = text[relabel_idx]
                    content.append("<image>")
                    images.append(_image)

                    # Put the model's output from previous step
                    if rag.get("keep_label_for_hot_image", False):
                        if isinstance(generated_label, list | tuple):
                            for gl in generated_label:
                                content.extend(_get_conv(gl, rag.get("prompt_for_sample", prompt)))
                                content.append(batched_contexts[ctx_id])

                            content = content[:-1]

                        else:
                            content.extend(
                                _get_conv(generated_label, rag.get("prompt_for_sample", prompt))
                            )

                    _rag_data.append(
                        {
                            "rag_data": content,
                            "images": images,
                        }
                    )

            else:
                raise NotImplementedError(f"Format {format} not implemented for RAG `pipe_from`")

            retrieved_data.append(_rag_data)

        return retrieved_data

    def generate_until_multi_round(self, requests: list[TaskInstance]) -> list[str]:
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
            tokens = self._tok_encode(x[0])
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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            if gen_kwargs.get("resize_input_image", None) is not None:
                for sublist_idx in range(len(batched_visuals)):
                    for img_idx in range(len(batched_visuals[sublist_idx])):
                        if not isinstance(batched_visuals[sublist_idx][img_idx], Image.Image):
                            continue

                        # Scale down, keeping aspect ratio, if the image is larger
                        if (
                            max(batched_visuals[sublist_idx][img_idx].size)
                            > gen_kwargs["resize_input_image"]
                        ):
                            batched_visuals[sublist_idx][img_idx].thumbnail(
                                (
                                    gen_kwargs["resize_input_image"],
                                    gen_kwargs["resize_input_image"],
                                ),
                                Image.LANCZOS,
                            )

            # Setup RAG for multi-round
            rag_all_steps = (gen_kwargs or {}).get("rag") or {}
            self._setup_rag(gen_kwargs, multi_step=True)

            # Setup mapping from round idx to actual step idx
            round_idx_to_actual_step = {}
            round_idx = 0
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

                # Get current round visual and context from doc_to_text function
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
                        _,  # batched_visuals, but we keep them for RAG. Just make sure to not
                        # put them in the prompt again
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
                            rag_messages = self._batch_rag(
                                batched_doc_id,
                                batched_contexts,
                                batched_visuals,
                                {"rag": rag},
                                task,
                                split,
                                round_idx,
                            )

                            if rag.get("cache_and_reuse", False):
                                global_rag_messages_by_round[round_idx] = copy.deepcopy(
                                    rag_messages
                                )

                        # Put the input images in the messages (which are a list of
                        # `PIL.Image`s in this case)
                        if rag.get("include_input", False):
                            rag_messages[0]["rag_data"].extend(["<images>"] * len(batched_visuals))
                            rag_messages[0]["images"].extend(sum(batched_visuals, []))

                        if rag.get("include_input_separately", False):
                            for k in rag_messages:
                                if len(rag_messages[k]["images"]) >= 1 and rag.get(
                                    "remove_input_if_match_num_samples", False
                                ):
                                    continue
                                rag_messages[k]["rag_data"].extend(
                                    ["<images>"] * len(batched_visuals[k])
                                )
                                rag_messages[k]["images"].extend(batched_visuals[k])

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

                            # Since `cached_answers` contains the same answers repeated for
                            # each item in the batch, we can just keep the first one (`[0]`)
                            cached_answers = cached_answers[0][: rag.get("num_samples")]

                            # Same for `cached_data`
                            cached_data = global_data_by_round[round_idx][0]

                            # Remove the cached items from the current rag messages,
                            # so that we only process the new ones
                            rag_messages[0] = rag_messages[0][rag.get("num_samples") :]

                        # Process each item in the batch separately
                        for i, _visuals in enumerate(rag_messages.values()):
                            _history = None
                            if rag.get("format", "") in (
                                "interleaved",
                                "image-answer-pairs",
                                "all-image-answer-pairs",
                            ):
                                if isinstance(_visuals, dict):
                                    _history = _visuals["rag_data"]
                                else:
                                    _history = ["\n".join(vis["rag_data"]) for vis in _visuals]

                            if isinstance(_visuals, dict):
                                _visuals = [[x.convert("RGB")] for x in _visuals["images"]]
                            elif isinstance(_visuals, list) and isinstance(_visuals[0], dict):
                                _visuals = [
                                    [x.convert("RGB") for x in vis["images"]] for vis in _visuals
                                ]
                            else:
                                raise ValueError(
                                    "Unexpected format for visuals in RAG separate queries."
                                )

                            # Construct the actual conversation for the model
                            _contexts = [batched_contexts[i]] * len(_visuals)
                            messages, _batched_visuals, image_tensor, convs = self._make_history(
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
                            rag["force_recreate_image_tensor"] = True
                            if rag.get("gen_batch_size", None) is not None:
                                # Process in smaller batches to avoid OOM
                                answers = []
                                for j in range(0, len(messages), rag["gen_batch_size"]):
                                    _start_req, _end_req, gen_kwargs_rag, resize_input_image = (
                                        None,
                                        None,
                                        None,
                                        None,
                                    )
                                    if "start_req" in gen_kwargs:
                                        _start_req = gen_kwargs.pop("start_req")
                                    if "end_req" in gen_kwargs:
                                        _end_req = gen_kwargs.pop("end_req")
                                    if "rag" in gen_kwargs:
                                        gen_kwargs_rag = gen_kwargs.pop("rag")
                                    if "resize_input_image" in gen_kwargs:
                                        resize_input_image = gen_kwargs.pop("resize_input_image")

                                    _inputs, _generation_output = self._generate(
                                        messages[j : j + rag["gen_batch_size"]],
                                        _batched_visuals[j : j + rag["gen_batch_size"]],
                                        image_tensor,
                                        gen_kwargs,
                                        rag,
                                    )
                                    answers.append(self._decode(_generation_output))

                                    if _start_req is not None:
                                        gen_kwargs["start_req"] = _start_req
                                    if _end_req is not None:
                                        gen_kwargs["end_req"] = _end_req
                                    if gen_kwargs_rag is not None:
                                        gen_kwargs["rag"] = gen_kwargs_rag
                                    if resize_input_image is not None:
                                        gen_kwargs["resize_input_image"] = resize_input_image

                                    del _inputs, _generation_output
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

                                answers = sum(answers, [])

                                # Just make sure they are None so that we don't accidentally
                                # use them later. We don't, but if we do, at least it should
                                # crash and we can trace it back
                                inputs, generation_output = None, None

                            # Process all at once
                            else:
                                _start_req, _end_req, gen_kwargs_rag, resize_input_image = (
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                                if "start_req" in gen_kwargs:
                                    _start_req = gen_kwargs.pop("start_req")
                                if "end_req" in gen_kwargs:
                                    _end_req = gen_kwargs.pop("end_req")
                                if "rag" in gen_kwargs:
                                    gen_kwargs_rag = gen_kwargs.pop("rag")
                                if "resize_input_image" in gen_kwargs:
                                    resize_input_image = gen_kwargs.pop("resize_input_image")

                                inputs, generation_output = self._generate(
                                    messages, _batched_visuals, image_tensor, gen_kwargs, rag
                                )
                                answers = self._decode(generation_output)

                                if _start_req is not None:
                                    gen_kwargs["start_req"] = _start_req
                                if _end_req is not None:
                                    gen_kwargs["end_req"] = _end_req
                                if gen_kwargs_rag is not None:
                                    gen_kwargs["rag"] = gen_kwargs_rag
                                if resize_input_image is not None:
                                    gen_kwargs["resize_input_image"] = resize_input_image

                                del inputs, generation_output
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                            # Cleanup
                            rag.pop("force_recreate_image_tensor", None)

                            # Post-process the answers and store them
                            for i, answer in enumerate(answers):
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

                            # Flatten visuals
                            _visuals = sum(_visuals, [])

                            # If we are using cached answers for the fixed items, we have to
                            # store them as if we generated them now. Since fixed images are
                            # the beginning of the batch, we prepend them to the ones we've
                            # just generated
                            if cached_answers is not None:
                                answers = cached_answers + answers
                            if cached_data is not None:
                                _visuals = cached_data + _visuals

                            answers_by_round[round_idx].append(answers)
                            data_by_round[round_idx].append(_visuals)

                        # If caching is enabled and we include input, prepare the
                        # answers and data accordingly
                        if rag.get("cache_and_reuse", False) and rag.get("include_input", False):
                            # Here, `len(rag_messages)` should be 1
                            answers_by_round[round_idx] = answers_by_round[round_idx] * len(
                                batched_contexts
                            )
                            data_by_round[round_idx] = data_by_round[round_idx] * len(
                                batched_contexts
                            )

                            # Now, for each list in the two lists, retain only the
                            # first N items and then the i-th one
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

                        # When caching and reusing results, reuse the first result
                        # for all items in the batch
                        if rag.get("cache_and_reuse", False):
                            # When the input is not included, i.e., when all items
                            # are *actually* the same, we can just reuse the cached
                            # answers faking answers for all items in the batch
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
                        question_input, _batched_visuals, image_tensor, convs = self._make_history(
                            gen_kwargs,
                            batched_contexts,
                            batched_visuals,
                            rag,
                            rag_messages,
                            history=last_round_info,
                            step_idx=round_idx,
                        )

                        _start_req, _end_req, gen_kwargs_rag, resize_input_image = (
                            None,
                            None,
                            None,
                            None,
                        )
                        if "start_req" in gen_kwargs:
                            _start_req = gen_kwargs.pop("start_req")
                        if "end_req" in gen_kwargs:
                            _end_req = gen_kwargs.pop("end_req")
                        if "rag" in gen_kwargs:
                            gen_kwargs_rag = gen_kwargs.pop("rag")
                        if "resize_input_image" in gen_kwargs:
                            resize_input_image = gen_kwargs.pop("resize_input_image")

                        images_per_request = [
                            len(x) if x is not None else 1 for x in _batched_visuals
                        ]  # For stats
                        _inputs, generation_output = self._generate(
                            question_input, _batched_visuals, image_tensor, gen_kwargs, rag
                        )
                        input_ids = _inputs["input_ids"]
                        attention_masks = _inputs["attention_masks"]

                        if _start_req is not None:
                            gen_kwargs["start_req"] = _start_req
                        if _end_req is not None:
                            gen_kwargs["end_req"] = _end_req
                        if gen_kwargs_rag is not None:
                            gen_kwargs["rag"] = gen_kwargs_rag
                        if resize_input_image is not None:
                            gen_kwargs["resize_input_image"] = resize_input_image

                        gen_metrics = self._loglikelihood(input_ids, generation_output)
                        context_tokens_count = [
                            attention_masks[idx].sum() for idx in range(len(batched_contexts))
                        ]

                        answers = self._decode(generation_output)

                        del _inputs, input_ids, attention_masks, generation_output
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Get clean answers and store them in the conversation history
                    for i, answer in enumerate(answers):
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

                        convs[i].messages[-1][
                            1
                        ] = answer  # replace assistant placeholder with the answer

                    answers_by_round[round_idx] = answers

                    batched_round_results.append(answers)
                    batched_round_info.append(convs)

                    if rag.get("store_memory", False):
                        self._retriever[0].store_memory(sum(batched_visuals, []), answers)

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
                _ans = TaskSingleOutput(
                    answer=ans,
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
                )
                res.append(_ans)
                self.cache_hook.add_partial(
                    "generate_until_multi_round", (context, gen_kwargs), ans
                )
                pbar.update(1)

        # Reorder this group of results back to original unsorted form
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


@register_model("llava-onevision-qwen2-7b-ov")
def llava_onevision_qwen2_7b_ov(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 7B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
    conv_template = "qwen_1_5"
    model = LLaVAOnevision(model_name_or_path, conv_template=conv_template, **model_kwargs)
    return model


@register_model("llava-onevision-qwen2-7b-si")
def llava_onevision_qwen2_7b_si(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 7B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-7b-si"
    conv_template = "qwen_1_5"
    model = LLaVAOnevision(model_name_or_path, conv_template=conv_template, **model_kwargs)
    return model


@register_model("llava-onevision-qwen2-0.5b-ov")
def llava_onevision_qwen2_0_5b_ov(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 0.5B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    conv_template = "qwen_1_5"
    model_name = "llava_qwen"
    model = LLaVAOnevision(
        model_name_or_path,
        conv_template=conv_template,
        model_name=model_name,
        **model_kwargs,
    )
    return model


@register_model("llava-onevision-qwen2-0.5b-si")
def llava_onevision_qwen2_0_5b_si(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 0.5B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-0.5b-si"
    conv_template = "qwen_1_5"
    model_name = "llava_qwen"
    model = LLaVAOnevision(
        model_name_or_path,
        conv_template=conv_template,
        model_name=model_name,
        **model_kwargs,
    )
    return model
