import copy
import os
import re
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils import ModelOutput

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

from ._phi3v_processor import Phi3VProcessor
from ._phi4_modeling import Phi4MMForCausalLM

__all__ = ["Phi3v"]

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


class Phi3v(Model):
    """Phi3v model.

    Args:
    ----
        model_name_or_path (str): The name or path of the pre-trained model to use. Defaults to
            "microsoft/Phi-3-vision-128k-instruct".
        attn_implementation (str, optional): The attention implementation to use. Defaults to
            "flash_attention_2" if `flash_attn` is installed, "eager" otherwise.
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        batch_size (int): The batch size to use for inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
        - https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/
        - https://github.com/microsoft/Phi-3CookBook

    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/Phi-3-vision-128k-instruct",
        attn_implementation: str | None = (
            "flash_attention_2" if utils.package_available("flash_attn") else "eager"
        ),
        use_cache: bool = True,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._attn_implementation = attn_implementation
        self._use_cache = use_cache

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU", "DEEPSPEED"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
            "torch_dtype": self.dtype,
            "_attn_implementation": self._attn_implementation,
        }
        processor_kwargs = {
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
        }

        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        if "Phi-4" in self._model_name_or_path:
            self._model = Phi4MMForCausalLM.from_pretrained(
                self._model_name_or_path, **model_kwargs
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name_or_path, **model_kwargs
            )
        self._model.eval()
        self._model = torch.compile(self._model, mode="max-autotune", fullgraph=True)
        # Using fixed version from pull request rather than AutoProcessor
        if "Phi-3" in self._model_name_or_path or "Phi-3.5" in self._model_name_or_path:
            self._processor = Phi3VProcessor.from_pretrained(
                self._model_name_or_path, **processor_kwargs
            )

        else:
            self._processor = AutoProcessor.from_pretrained(
                self._model_name_or_path, **processor_kwargs
            )
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer

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
            multi_step (bool): Whether to set up for multi-step RAG. Defaults to False.

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
    ) -> dict | None:
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
                    _rag_data.append(elem.get("text", ""))
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
            gen_kwargs, images, prepare=prepare, result_callback=result_callback, doc_ids=doc_ids
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        result = list(cast(list[dict[str, list]], result))

        return result

    def _prepare_image_tokens(
        self, batched_visuals: list[Image.Image], start_counter: int, return_as_list: bool = False
    ) -> tuple[str | list[str], int, int]:
        """Prepare image tokens for the model.

        Args:
        ----
            batched_visuals (list[Image.Image]): List of images to prepare tokens for.
            start_counter (int): Starting counter for image tokens.
            return_as_list (bool): Whether to return image tokens as a list. Defaults to False.

        """
        counter = start_counter

        if return_as_list:
            image_tokens = []
            for _ in range(len(batched_visuals)):
                image_tokens.append(f"<|image_{counter+1}|>")
                counter += 1
        else:
            image_tokens = ""
            for _ in range(len(batched_visuals)):
                image_tokens += f"<|image_{counter+1}|>\n"
                counter += 1

        return image_tokens, counter, start_counter

    def _compose_rag_message(self, rag_message: list[str], rag_image_tokens: list[str]) -> str:
        """Compose the RAG message by replacing <images> placeholders with actual image tokens.

        Args:
        ----
            rag_message (list[str]): The RAG message with placeholders.
            rag_image_tokens (list[str]): The image tokens to replace the placeholders.

        """
        composed_message = []

        image_idx = 0
        for x in rag_message:
            if x != "<images>":
                composed_message.append(x)
            else:
                if image_idx < len(rag_image_tokens):
                    composed_message.append(rag_image_tokens[image_idx])
                    image_idx += 1
                else:
                    log.warning(
                        "Not enough images to replace <images> placeholder in RAG message."
                    )

        return "\n".join(composed_message)

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
                if cache_and_reuse:
                    break

        # Batch retrieve data
        payload = [{"doc_id": ids, **self.task_dict[task][split][ids]} for ids in batched_doc_id]
        retrieved_data = self._retrieve(
            gen_kwargs, list(batched_visuals_for_rag.values()), payload, step_idx=step_idx
        )
        for k, v in zip(batched_visuals_for_rag.keys(), retrieved_data, strict=True):
            rag_messages[k] = v

        return rag_messages

    def _generate(
        self, context: list, batched_visuals: list, gen_kwargs: dict, rag: dict
    ) -> tuple[dict, ModelOutput]:
        """Generate model outputs for the given messages.

        Args:
        ----
            context (list): A list of messages.
            batched_visuals (list): A list of visual inputs for the batch.
            gen_kwargs (dict): Generation keyword arguments.
            rag (dict): RAG configuration dictionary.

        """
        _batched_visuals = _flatten_list(batched_visuals)
        _batched_visuals = [x.convert("RGB") for x in _batched_visuals]

        input_ids = self.processor(text=context, images=_batched_visuals, return_tensors="pt").to(
            self.device, self.model.dtype
        )

        # Setting default parameters.
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        # Generate answer
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eod_id
        )
        with torch.inference_mode():
            generation_output = self.model.generate(
                **input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self._use_cache,
                return_dict_in_generate=True,
                output_scores=True,
            )

        return input_ids, generation_output

    def _decode(self, inputs: dict, generation_output: ModelOutput) -> list[str]:
        """Decode the generated output into a list of strings.

        Args:
        ----
            inputs (dict): The input dictionary containing input IDs.
            generation_output (ModelOutput): The output from the model generation.

        """
        generated_ids = generation_output.sequences

        generate_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return responses

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
    ) -> tuple[list, list, list]:
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

        image_counter = 0
        _batched_visuals = []
        all_messages = []
        for i in range(len(batched_contexts)):
            _batched_visual_to_add = batched_visuals[i]
            # Disable image after first step when RAG is not enabled
            start_batched_visual = 0
            if not rag_enabled and step_idx is not None and step_idx > 0:
                start_batched_visual = 1
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                    if rag.get("system_prompt") is None
                    else rag.get("system_prompt"),
                }
            ]

            if history is not None and history[i] is not None:
                if append_history:
                    if isinstance(history[i], list):
                        for msg in history[i]:
                            messages.append({"role": "user", "content": msg})
                    elif isinstance(history[i], str):
                        messages.append({"role": "user", "content": history[i]})
                    else:
                        messages.extend(history[i])
                else:
                    messages = copy.deepcopy(history[i])

                initial_image_counter = image_counter
                messages, image_counter = self._adjust_image_tokens(messages, image_counter)
                added_images = image_counter - initial_image_counter
                start_batched_visual += added_images

            rag_message = None
            if rag_enabled:
                if i in rag_messages:
                    rag_message = rag_messages[i]
                else:
                    rag_message = self._retrieve(
                        gen_kwargs, batched_visuals[i], step_idx=step_idx
                    )[0]
                # Get the actual message and the images
                rag_message, rag_images = rag_message["rag_data"], rag_message["images"]

            # Build the prompt with <image> tokens and RAG content, if any
            if "<image>" in batched_contexts[i]:
                query = "" + batched_contexts[i]
                img_placeholder_count = 1
                while "<image>" in query:
                    query = query.replace("<image>", f"<|image_{img_placeholder_count}|>", 1)
                    img_placeholder_count += 1
            else:
                # No RAG: just use the vanilla model. To ensure this doesn't break
                # when piping data, we can disable adding image tokens by setting the
                # `add_image_tokens_when_no_rag` flag to False
                if not rag_enabled or rag_message is None:
                    image_tokens, image_counter, _ = self._prepare_image_tokens(
                        batched_visuals[i][start_batched_visual:], image_counter
                    )  # noqa: E501

                    if include_target_classes and include_target_classes_position == "pre-query":
                        query = (
                            image_tokens
                            + "\n"
                            + target_classes_prompt.format(target_classes_str)
                            + "\n"
                            + batched_contexts[i]
                        )
                    elif (
                        include_target_classes and include_target_classes_position == "end-of-ctx"
                    ):
                        query = (
                            image_tokens
                            + "\n"
                            + batched_contexts[i]
                            + "\n"
                            + target_classes_prompt.format(target_classes_str)
                        )
                    else:
                        query = image_tokens + batched_contexts[i]

                # RAG is enabled and there is actual retrieved content
                elif rag_enabled and rag_message is not None:
                    context = copy.deepcopy(batched_contexts[i])
                    if include_target_classes and include_target_classes_position == "pre-query":
                        context = target_classes_prompt.format(target_classes_str) + "\n" + context
                    elif (
                        include_target_classes and include_target_classes_position == "end-of-ctx"
                    ):
                        context = context + target_classes_prompt.format(target_classes_str)

                    # Retrieved data goes at the beginning of the context
                    if rag_position == "pre-sample":
                        rag_image_tokens, image_counter, _ = self._prepare_image_tokens(
                            rag_images, image_counter, return_as_list=True
                        )
                        image_tokens, image_counter, _ = self._prepare_image_tokens(
                            batched_visuals[i], image_counter
                        )

                        rag_message = self._compose_rag_message(rag_message, rag_image_tokens)

                        if len(rag_images) > 0:
                            _batched_visual_to_add = rag_images + _batched_visual_to_add

                        query = "".join(
                            [
                                rag_message,
                                "\n\n",
                                image_tokens,
                                context,
                            ]
                        )

                    # Retrieved data goes after the image, before the query
                    if rag_position == "post-sample":
                        image_tokens, image_counter, _ = self._prepare_image_tokens(
                            batched_visuals[i], image_counter
                        )
                        rag_image_tokens, image_counter, _ = self._prepare_image_tokens(
                            rag_images, image_counter
                        )
                        rag_message = self._compose_rag_message(rag_message, rag_image_tokens)

                        if len(rag_images) > 0:
                            _batched_visual_to_add = _batched_visual_to_add + rag_images

                        query = "".join(
                            [
                                image_tokens,
                                "\n",
                                rag_message,
                                "\n\n",
                                context,
                            ]
                        )

                    # Retrieved data goes at the end of the context
                    if rag_position == "post-sample-and-query":
                        image_tokens, image_counter, _ = self._prepare_image_tokens(
                            batched_visuals[i], image_counter
                        )
                        rag_image_tokens, image_counter, _ = self._prepare_image_tokens(
                            rag_images, image_counter
                        )
                        rag_message = self._compose_rag_message(rag_message, rag_image_tokens)

                        if len(rag_images) > 0:
                            _batched_visual_to_add = _batched_visual_to_add + rag_images

                        query = "".join(
                            [
                                image_tokens,
                                "\n",
                                context,
                                "\n\n",
                                rag_message,
                            ]
                        )

                else:
                    if include_target_classes and include_target_classes_position == "pre-query":
                        query = (
                            target_classes_prompt.format(target_classes_str)
                            + "\n"
                            + batched_contexts[i]
                        )
                    elif (
                        include_target_classes and include_target_classes_position == "end-of-ctx"
                    ):
                        query = (
                            batched_contexts[i]
                            + "\n"
                            + target_classes_prompt.format(target_classes_str)
                        )
                    else:
                        query = "" + batched_contexts[i]

            _batched_visuals.append(_batched_visual_to_add)

            messages.append({"role": "user", "content": query})
            all_messages.append(messages)
            batched_contexts[i] = self._tokenize(messages)

        return all_messages, batched_contexts, _batched_visuals

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

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            if isinstance(batched_contexts, tuple):
                batched_contexts = list(batched_contexts)

            # RAG setup (guard-style, avoids deep nesting later)
            rag_enabled = self._is_rag_enabled(gen_kwargs)
            rag = (gen_kwargs or {}).get("rag") or {}
            rag["rag_enabled"] = rag_enabled
            if rag_enabled:
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

            messages, batched_contexts, _batched_visuals = self._make_history(
                gen_kwargs,
                batched_contexts,
                batched_visuals,
                rag,
                rag_messages,
            )

            # Moved here so that the number of visuals corresponds to the number of contexts
            # and the code that inserts <image> tokens correctly sees the number of images
            # associated to each context
            images_per_request = [len(x) for x in _batched_visuals]

            input_ids, generation_output = self._generate(
                batched_contexts, _batched_visuals, gen_kwargs, rag
            )
            response = self._decode(input_ids, generation_output)
            gen_metrics = self._loglikelihood(input_ids, generation_output)

            # Cleanup RAG keys
            rag.pop("doc_to_target", None)
            rag.pop("doc_to_visual", None)
            rag.pop("test_docs", None)

            for idx, (ans, context) in enumerate(zip(response, batched_contexts, strict=True)):
                _ans = TaskSingleOutput(
                    answer=ans,
                    context=context,
                    context_tokens_count=input_ids["attention_mask"][idx].sum(),
                    num_images=images_per_request[idx],
                    loglikelihood=gen_metrics["loglikelihood"][idx].item(),
                    perplexity=gen_metrics["perplexity"][idx].item(),
                )
                res.append(_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

            del input_ids, generation_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            answers_by_round (dict): Anserws from previous rounds.
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
                raise NotImplementedError("Not implemented.")

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

                    # Add the conversation to the results
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

    def _adjust_image_tokens(
        self, messages: list, image_counter: int = 0, batched: bool = False
    ) -> tuple[list, int]:
        """Adjust image tokens in the messages to ensure they are unique across the conversation.

        Args:
        ----
            messages (list): The conversations to process.
            image_counter (int): Start count for image tokens.
            batched (bool): Whether the input is batched.

        """
        if batched:
            results = []
            for msg in messages:
                adjusted_msgs, image_counter = self._adjust_image_tokens(
                    msg, image_counter=image_counter, batched=False
                )
                results.append((adjusted_msgs, image_counter))
            return results

        for msg_idx, msg in enumerate(messages):
            if "content" in msg and isinstance(msg["content"], str):
                content = msg["content"]

                # Also replace standard <image> placeholders. This is required when
                # processing RAG messages. Using id 0, which is illegal, as we update
                # it later anyway. This ensures that if something goes wrong, the forward
                # method will raise an error due to invalid image token.
                if "<image>" in content:
                    content = content.replace("<image>", "<|image_0|>")

                img_placeholders = re.finditer(r"<\|image_(\d+)\|>", content)
                placeholders = [(match.start(), match.group(1)) for match in img_placeholders]

                if not placeholders:
                    messages[msg_idx]["content"] = content
                    continue

                placeholders.sort(key=lambda x: x[0])

                new_content = ""
                last_pos = 0
                for start_pos, placeholder_id in placeholders:
                    # Append text before the placeholder
                    new_content += content[last_pos:start_pos]
                    # Append new unique placeholder
                    new_content += f"<|image_{image_counter+1}|>"
                    image_counter += 1
                    last_pos = start_pos + len(f"<|image_{placeholder_id}|>")

                # Append remaining content
                new_content += content[last_pos:]

                # Step 5: Update the message content
                messages[msg_idx]["content"] = new_content

        return messages, image_counter

    def _tokenize(self, messages: list) -> str:
        """Tokenize the messages using the tokenizer's chat template.

        Args:
        ----
            messages (list): The conversation messages to tokenize.

        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            if isinstance(batched_contexts, tuple):
                batched_contexts = list(batched_contexts)

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
                        _,  # batched_visuals, but we keep them for RAG. Just make sure to
                        # not put them in the prompt again
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
                            messages, _batched_contexts, _batched_visuals = self._make_history(
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

                                for j in range(0, len(_batched_contexts), rag["gen_batch_size"]):
                                    updated_messages = list(
                                        zip(
                                            *self._adjust_image_tokens(
                                                messages[j : j + rag["gen_batch_size"]],
                                                batched=True,
                                            ),
                                            strict=False,
                                        )
                                    )[0]
                                    _batched_contexts[
                                        j : j + rag["gen_batch_size"]
                                    ] = self._tokenize(updated_messages)

                                    _inputs, _generation_output = self._generate(
                                        _batched_contexts[j : j + rag["gen_batch_size"]],
                                        _batched_visuals[j : j + rag["gen_batch_size"]],
                                        gen_kwargs,
                                        rag,
                                    )

                                    answers.append(self._decode(_inputs, _generation_output))
                                    del _inputs, _generation_output
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

                                answers = _flatten_list(answers)

                                # Just make sure they are None so that we don't accidentally
                                # use them later. We don't, but if we do, at least it should
                                # crash and we can trace it back
                                _, generation_output = None, None

                            # Process all at once
                            else:
                                input_ids, generation_output = self._generate(
                                    _batched_contexts, _batched_visuals, gen_kwargs, rag
                                )
                                answers = self._decode(input_ids, generation_output)

                                del input_ids, generation_output
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
                                        if len(temp_answer) > 0 and temp_answer != "\n":
                                            answer = temp_answer
                                            break

                                    # Fallback in case all labels were empty after cleaning
                                    if isinstance(answer, list):
                                        answer = answer[0]

                                answers[i] = answer

                            # Flatten visuals
                            _visuals = _flatten_list(_visuals)

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

                        # If caching is enabled and we include input, prepare
                        # the answers and data accordingly
                        if rag.get("cache_and_reuse", False) and rag.get("include_input", False):
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
                        if passthrough_answers is None:
                            raise ValueError(
                                "`passthrough_answers` cannot be None "
                                "when `rag['passthrough']` is `True`"
                            )
                        answers = passthrough_answers
                        answers_by_round[round_idx] = answers
                        generation_output = None
                        messages = [[]] * len(answers)

                    else:
                        if gen_kwargs.get("resize_input_image", None) is not None:
                            for img in _flatten_list(batched_visuals):
                                if not isinstance(img, Image.Image):
                                    continue

                                # Scale down, keeping aspect ratio, if the image is larger
                                if max(img.size) > gen_kwargs["resize_input_image"]:
                                    img.thumbnail(
                                        (
                                            gen_kwargs["resize_input_image"],
                                            gen_kwargs["resize_input_image"],
                                        ),
                                        Image.LANCZOS,
                                    )

                        messages, batched_contexts, _batched_visuals = self._make_history(
                            gen_kwargs,
                            batched_contexts,
                            batched_visuals,
                            rag,
                            rag_messages,
                            history=last_round_info,
                            step_idx=round_idx,
                        )

                        updated_messages = list(
                            zip(*self._adjust_image_tokens(messages, batched=True), strict=False)
                        )[0]
                        batched_contexts = self._tokenize(updated_messages)

                        images_per_request = [
                            len(x) if x is not None else 1 for x in _batched_visuals
                        ]

                        input_ids, generation_output = self._generate(
                            batched_contexts, _batched_visuals, gen_kwargs, rag
                        )
                        answers = self._decode(input_ids, generation_output)
                        gen_metrics = self._loglikelihood(input_ids, generation_output)

                        context_tokens_count = [
                            input_ids["attention_mask"][idx].sum()
                            for idx in range(len(batched_contexts))
                        ]

                        del input_ids, generation_output, _batched_visuals
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
                        messages[i].append({"role": "assistant", "content": answer})

                    batched_round_results.append(answers)
                    batched_round_info.append(messages)

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


@register_model("phi3v")
def phi3v(**model_kwargs) -> Model:
    """Load the Phi3V model with 4B params."""
    model_name_or_path = "microsoft/Phi-3-vision-128k-instruct"
    model = Phi3v(model_name_or_path, **model_kwargs)
    return model


@register_model("phi3.5v")
def phi3_5v(**model_kwargs) -> Model:
    """Load the Phi3.5V model with 4B params."""
    model_name_or_path = "microsoft/Phi-3.5-vision-instruct"
    model = Phi3v(model_name_or_path, **model_kwargs)
    return model


@register_model("phi4")
def phi4v(**model_kwargs) -> Model:
    """Load the Phi4 multimodal model with 6B params."""
    model_name_or_path = "microsoft/Phi-4-multimodal-instruct"
    model = Phi3v(model_name_or_path, **model_kwargs)
    return model
