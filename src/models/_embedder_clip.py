import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src import utils
from src.data.tasks import TaskInstance
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model

__all__ = ["EmbedderCLIP"]

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


class EmbedderCLIP(Model):
    """CLIP Model, used to embed stuff.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "Qwen/Qwen2-VL-7B-Instruct".
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether to use flash attention 2. Default to False.
        max_pixels (int): The max number of pixels in an image. Defaults to 12'845'056.
        min_pixels (int): The min number of pixels in an image. Defaults to 3'316.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        use_cache: bool = True,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._use_cache = use_cache
        self.batch_size_per_gpu = batch_size

        db_root = os.getenv("RAG_DATABASE_ROOT")
        if not db_root:
            raise ValueError("RAG_DATABASE_ROOT is not set")
        self._db_root = Path(db_root)

        self._transform: T.Compose | None = None

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path, device_map=self.device_map
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self._model_name_or_path)

        if self._model_name_or_path in [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]:
            channel_stats = dict(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            self._transform = T.Compose(
                [
                    T.Resize(size=(224, 224), interpolation=Image.Resampling.BICUBIC),
                    T.CenterCrop(224),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(**channel_stats),
                ]
            )

    def loglikelihood(self, requests: list[TaskInstance]) -> list[tuple[float, bool]]:
        """Compute the log-likelihood of the given requests.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, target). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError

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
            return 1, x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__
        dataset_name = configurable_task.task_name.split("_embed")[0]

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        doc_ids = []
        visuals = []
        targets = []

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
            batched_visuals = _flatten_list(batched_visuals)

            doc_ids.extend(batched_doc_id)
            visuals.extend([self.task_dict[task][split][ids]["visual"] for ids in batched_doc_id])
            targets.extend([self.task_dict[task][split][ids]["target"] for ids in batched_doc_id])

            if self._transform is not None:
                queries = torch.stack(
                    [
                        self._transform(
                            T.functional.to_image(image).to(self._device, non_blocking=True)
                        )
                        for image in batched_visuals
                    ],
                    dim=0,
                )
                inputs = {
                    "pixel_values": queries,
                }
            else:
                inputs = self._processor(images=batched_visuals, return_tensors="pt").to(
                    self._device
                )

            with torch.no_grad():
                embeddings = self._model.get_image_features(**inputs)
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            res.append(embeddings.to("cpu"))
            pbar.update(len(embeddings))

        res = torch.cat(res, dim=0)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)
        res = torch.stack(res, dim=0)
        doc_ids = reordered.get_original(doc_ids)
        visuals = reordered.get_original(visuals)
        targets = reordered.get_original(targets)

        metadata = {}
        for i, (ids, vis, tgt) in enumerate(zip(doc_ids, visuals, targets, strict=False)):
            metadata[i] = {
                "doc_id": ids,
                "visual": vis,
                "target": tgt,
            }

        model_nicename = self._model_name_or_path.replace("/", "-")
        save_path = self._db_root / dataset_name / split
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(res, save_path / f"{model_nicename}.pt")

        with open(save_path / f"{model_nicename}_metadata.json", "w") as f:
            json.dump(metadata, f)

        pbar.close()

        out = f"Embeddings saved to {save_path}. Exiting"
        log.info(out)
        exit()

        return res

    def generate_until_multi_round(self, requests: list[TaskInstance]) -> list[str]:
        """Generate greedily until a stopping sequence with multi-round generation.

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
        raise NotImplementedError


@register_model("clip-vit-b32-openai")
def clip_vit_b32_openai(**model_kwargs) -> Model:
    """Load the CLIP ViT B/32 model from OpenAI."""
    model_name_or_path = "openai/clip-vit-base-patch32"
    model = EmbedderCLIP(model_name_or_path, **model_kwargs)
    return model


@register_model("clip-vit-b16-openai")
def clip_vit_b16_openai(**model_kwargs) -> Model:
    """Load the CLIP ViT B/16 model from OpenAI."""
    model_name_or_path = "openai/clip-vit-base-patch16"
    model = EmbedderCLIP(model_name_or_path, **model_kwargs)
    return model


@register_model("clip-vit-l14-openai")
def clip_vit_l14_openai(**model_kwargs) -> Model:
    """Load the CLIP ViT L/14 model from OpenAI."""
    model_name_or_path = "openai/clip-vit-large-patch14"
    model = EmbedderCLIP(model_name_or_path, **model_kwargs)
    return model
