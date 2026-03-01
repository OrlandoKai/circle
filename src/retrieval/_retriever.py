# pytype: skip-file

import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src import utils
from src.retrieval import (
    DynamicRetrievalTensorDatabase,
    RetrievalDatabase,
    RetrievalTensorDatabase,
    get_images_by_key,
)

__all__ = ["Retriever"]

log = utils.get_logger(__name__, rank_zero_only=True)


class Retriever:
    """Retriever to retrieve similar images from a retrieval database."""

    def __init__(
        self,
        path: str | Path,
        format: str = "faiss",
        model_name: str = "openai/clip-vit-base-patch32",
        use_flash_attention_2: bool | None = utils.package_available("flash_attn"),
        dtype: str | torch.dtype = "float16",
        device_map: str = "auto",
        few_shot: int | None = None,
    ) -> None:
        """Retrieve similar images from a retrieval database.

        Args:
        ----
            path (str | Path): Path to the retrieval database.
            format (str): Format of the retrieval database. Either 'faiss', 'pt' or 'dynamic_pt'.
            model_name (str): Name of the pre-trained model to use.
            use_flash_attention_2 (bool | None): Whether to use flash attention 2 if available.
            dtype (str | torch.dtype): Data type to use for the model.
            device_map (str): Device map to use for the model. Either 'auto' or
                'cpu'. If 'auto', the model will be loaded on GPU if available.
            few_shot (int | None): If specified, restrict the database to few-shot
                examples per class. Works only for 'pt' format.

        """
        self._path = path
        self._use_flash_attention_2 = use_flash_attention_2

        assert format in ["faiss", "pt", "dynamic_pt"], "format is not valid"

        self._device = "cuda" if torch.cuda.is_available() and device_map == "auto" else "cpu"

        self._format = format
        if format == "faiss":
            self._database = RetrievalDatabase(path)
        elif format == "pt":
            self._database = RetrievalTensorDatabase(path, device=self._device)
            if few_shot is not None:
                self._database.few_shot(few_shot)
        elif format == "dynamic_pt":
            self._database = DynamicRetrievalTensorDatabase(device=self._device)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self._model = AutoModel.from_pretrained(model_name, device_map=device_map)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._interface = "clip"
        self._model.eval()

        self._vocab_transform = lambda x: x

        self._transform = None
        if model_name in [
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

    def set_vocab_transform(self, rag: dict) -> None:
        """Set the vocabulary transform function based on the RAG configuration.

        Args:
        ----
            rag (dict): RAG configuration dictionary.

        """
        if rag.get("format", "list-captions") == "cased":
            self._vocab_transform = utils.default_vocabulary_transforms()

    @torch.no_grad()
    def store_memory(
        self,
        images: Image.Image | list[Image.Image],
        labels: str | list[str],
    ) -> None:
        """Store the given image(s) in the retrieval database with the corresponding label(s).

        Args:
        ----
            images (Image.Image | list[Image.Image]): The image(s) to store.
            labels (str | list[str]): The label(s) corresponding to the image(s).

        """
        assert self._format in [
            "dynamic_pt"
        ], "`store_memory` is only supported for the `dynamic_pt` format."

        if not isinstance(images, list):
            images = [images]
        if not isinstance(labels, list):
            labels = [labels]

        assert len(images) == len(labels), "Number of images and labels must be the same."

        # Embed images so that they can be stored
        embeddings = self._embed_image_queries(images)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        self._database.add_data(embeddings, list(zip(images, labels, strict=True)))

    @torch.no_grad()
    def _embed_image_queries(self, queries: list[Image.Image]) -> torch.Tensor:
        """Embed image queries into the joint embedding space. Returns unnormalized embeddings.

        Args:
        ----
            queries (list[Image.Image]): List of input images.

        """
        if self._transform is not None:
            queries = torch.stack(
                [
                    self._transform(
                        T.functional.to_image(image).to(self._device, non_blocking=True)
                    )
                    for image in queries
                ],
                dim=0,
            )
            inputs = {
                "pixel_values": queries,
            }
        else:
            inputs = self._processor(images=queries, return_tensors="pt").to(self._device)

        embeddings = self._model.get_image_features(**inputs)

        return embeddings

    def _get_target_classes(self, rag: dict) -> list[dict]:
        """Get target classes from the RAG configuration.

        Args:
        ----
            rag (dict): RAG configuration dictionary.

        """
        doc_to_target = rag.get("doc_to_target")
        test_docs = rag.get("test_docs")
        if doc_to_target is None or test_docs is None:
            raise ValueError(
                "When using 'in-domain' search modality, 'doc_to_target' and 'test_docs' must be provided in rag configuration."  # noqa: E501
            )

        targets = sorted(set(test_docs()["target"]))

        results_set = [
            {
                "image_path": target,  # Dummy, not actually used
                "caption": doc_to_target({"target": target}),
            }
            for target in targets
        ]

        return results_set

    @torch.no_grad()
    def retrieve(
        self,
        queries: str | list[str] | Image.Image | list[Image.Image],
        input_type: str = "image",
        search_modality: str = "text",
        num_samples: int = 10,
    ) -> list[dict]:
        """Retrieve similar images from the database.

        Args:
        ----
            queries (str | list[str] | Image.Image | list[Image.Image]): Input queries
            input_type (str): Type of the input queries. Either 'image' or 'text'.
            search_modality (str): Modality to search in the database. Either 'image'
                or 'text'.
            num_samples (int): Number of samples to retrieve.

        """
        assert input_type in ["image", "text"], "input_type must be either 'image' or 'text'"
        assert search_modality in [
            "image",
            "text",
        ], "search_modality must be either 'image' or 'text'"

        # Ensure the input is a list
        if not isinstance(queries, list):
            queries = [queries]

        if input_type == "image":
            for i in range(len(queries)):
                if isinstance(queries[i], str):
                    queries[i] = Image.open(queries[i]).convert("RGB")
            embeddings = self._embed_image_queries(queries)

        elif input_type == "text":
            inputs = self._processor(
                text=queries, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            embeddings = self._model.get_text_features(**inputs)

        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if self._format == "faiss":
            embeddings = embeddings.cpu().numpy()

        # Retrieve similar images from the database
        results = self._database.query(
            embeddings, modality=search_modality, num_samples=num_samples
        )

        return results

    def _prepare_context_with_images(
        self, prompts: list[str], retrieved_images: list, rag_text: str, format: str
    ) -> list[dict[str, Any]]:
        """Prepare the context with images and text for RAG.

        Args:
        ----
            prompts (list[str]): List of prompt strings split by the <images> token.
            retrieved_images (list): List of retrieved image dictionaries.
            rag_text (str): Text to include in the context.
            format (str): Format of the RAG context.

        """
        rag_data = []
        if len(prompts) > 0 and len(prompts[0]) > 0:
            rag_data.append({"type": "text", "text": prompts[0]})

        rag_data.extend(retrieved_images)

        if format != "no-text" and len(prompts[1]) > 0:
            rag_data.append({"type": "text", "text": prompts[1].format(rag_text)})

        return rag_data

    def retrieve_and_prepare(
        self,
        gen_kwargs: dict,
        images: list[Image.Image],
        prepare: Callable | None = None,
        result_callback: Callable | None = None,
        doc_ids: dict | list | None = None,
    ) -> list[list[dict[str, Any]]] | list[dict[str, Any]] | None:
        """Retrieve relevant data for each image and prepare it for generation.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments. It must contain a 'rag' object
            images (list[Image.Image]): List of input images.
            prepare (Callable, optional): Function to prepare the retrieved images for generation.
            result_callback (Callable, optional): Function to process the final retrieved data.
            doc_ids (dict | list | None, optional): Document IDs to retrieve. Defaults to None.

        """
        rag = (gen_kwargs or {}).get("rag") or {}

        # Input is always images. Search is conducted by comparing with
        # either text embeddings or image embeddings
        search_modality = rag.get("search_modality", "text")

        # From the retrieved data, determine the type of data to grab:
        # the retrieved data itself or the corresponding image/text
        grab_data_type = rag.get("grab_data_type", search_modality)

        # Randomly sample contexts from the database
        if search_modality == "random":
            if self._format == "faiss":
                indices = list(range(len(self._database._metadata_provider)))

                results_set = []
                for _ in range(len(images)):
                    sampled_idxs = random.sample(indices, k=rag.get("num_samples", 10))
                    retrieved = self._database._metadata_provider[sampled_idxs]
                    results_set.append(retrieved)

            elif self._format in ["pt", "dynamic_pt"]:
                doc_to_target = rag.get("doc_to_target")
                doc_to_visual = rag.get("doc_to_visual")

                results_set = []
                if hasattr(self._database._metadata_provider, "keys"):
                    keys = list(self._database._metadata_provider.keys())
                else:
                    # Fallback, should never happen if the database is properly implemented
                    keys = []

                if len(keys) == 0:
                    rag_data_set = [[] for _ in range(len(images))]
                    if result_callback is not None:
                        rag_data_set = [
                            result_callback(rag_data, images=None) for rag_data in rag_data_set
                        ]
                    return rag_data_set

                for _ in range(len(images)):
                    sampled_keys = random.sample(keys, min(rag.get("num_samples", 10), len(keys)))
                    retrieved = [self._database._metadata_provider[k] for k in sampled_keys]
                    results_set.append(retrieved)

                _in_domain_images = {}
                counter = 0
                for results in results_set:
                    for result in results:
                        if not isinstance(result, dict):
                            raise ValueError(
                                "Each retrieved result must be a dictionary containing "
                                "at least the keys 'target' and 'visual'."
                            )

                        if self._format == "dynamic_pt":
                            result["caption"] = result["target"]
                            result["target"] += f"_{counter}"
                            result["image_path"] = result["target"]
                            _in_domain_images[result["target"]] = result["visual"]
                        else:
                            result["caption"] = doc_to_target(result)
                            result["target"] += f"_{counter}"
                            result["image_path"] = result["target"]
                            _in_domain_images[result["target"]] = doc_to_visual(result)[0]
                        counter += 1

        # Retrieve in-domain images (random, from test set)
        elif search_modality == "in-domain":
            doc_to_target = rag.get("doc_to_target")
            doc_to_visual = rag.get("doc_to_visual")
            test_docs = rag.get("test_docs")
            if doc_to_target is None or doc_to_visual is None or test_docs is None:
                raise ValueError(
                    "When using 'in-domain' search modality, 'doc_to_target', 'doc_to_visual' and 'test_docs' must be provided in rag configuration."  # noqa: E501
                )

            results_set = []
            _in_domain_images = {}
            for sample in test_docs():
                target = doc_to_target(sample)
                if target in _in_domain_images:
                    continue

                _in_domain_images[target] = doc_to_visual(sample)[0]

                results_set.append(
                    {
                        "image_path": target,
                        "caption": target,
                    }
                )

            results_set = [results_set] * len(images)

        elif search_modality == "in-domain-rag":
            doc_to_target = rag.get("doc_to_target")
            doc_to_visual = rag.get("doc_to_visual")
            if doc_to_target is None or doc_to_visual is None:
                raise ValueError(
                    "When using 'in-domain-rag' search modality, 'doc_to_target' and 'doc_to_visual' must be provided in rag configuration."  # noqa: E501
                )

            # Ensure that it doesn't fail when the database is empty
            if self._format == "dynamic_pt" and len(self._database._metadata_provider) == 0:
                rag_data_set = [[] for _ in range(len(images))]
                if result_callback is not None:
                    rag_data_set = [
                        result_callback(rag_data, images=None) for rag_data in rag_data_set
                    ]
                return rag_data_set

            results_set = self.retrieve(
                images,
                input_type="image",
                num_samples=rag.get("num_samples", 10),
            )

            if rag.get("majority_voting", False):
                for result_idx in range(len(results_set)):
                    labels = [x.get("target", "").strip() for x in results_set[result_idx]]

                    # Keep only results in results_set[result_idx] that have the most common label
                    if len(labels) == 0:
                        continue

                    # Use counter
                    label_counts = Counter(labels)
                    most_common_label, _ = label_counts.most_common(1)[0]
                    results_set[result_idx] = [
                        x
                        for x in results_set[result_idx]
                        if x.get("target", "").strip() == most_common_label
                    ]

            _in_domain_images = {}
            counter = 0
            for results in results_set:
                for result in results:
                    if self._format == "dynamic_pt":
                        result["caption"] = result["target"]
                        result["target"] += f"_{counter}"
                        result["image_path"] = result["target"]
                        _in_domain_images[result["target"]] = result["visual"]
                    else:
                        result["caption"] = doc_to_target(result)
                        result["target"] += f"_{counter}"
                        result["image_path"] = result["target"]

                        if grab_data_type == "image":
                            _in_domain_images[result["target"]] = doc_to_visual(result)[0]
                    counter += 1

        # Retrieve target classes (text-only)
        elif search_modality == "target-classes":
            results_set = self._get_target_classes(rag)
            # Replicate for each input image
            results_set = [results_set] * len(images)

        # Standard retrieval based on similarity
        else:
            results_set = self.retrieve(
                images,
                input_type="image",
                search_modality=search_modality,
                num_samples=rag.get("num_samples", 10),
            )

        # Shuffle the data?
        if rag.get("shuffle", False):
            for results in results_set:
                random.shuffle(results)

        if results_set is None:
            return None

        # ============================ #
        # Use the retrieved image data #
        # ============================ #
        if grab_data_type == "image":
            prompts = rag.get("prompt", "")
            format = rag.get("format", "list-captions")
            limit = rag.get("limit", None)
            caption_key = rag.get("caption_key", "caption")

            assert "<images>" in prompts, "Prompt must contain the <images> token."
            prompts = prompts.split("<images>")

            # Collect all image keys to retrieve in a single batch
            all_images_keys = []
            batch_idxs = []
            for batch_idx, results in enumerate(results_set):
                if not isinstance(results, list):
                    raise ValueError(
                        "Each item in results_set must be a list of retrieved results."
                    )

                image_keys = [x.get("image_path") for x in results]
                # Enforce a limit on the maximum number of images retrieved
                if limit is not None:
                    image_keys = image_keys[:limit]

                all_images_keys.extend(image_keys)
                batch_idxs.extend([batch_idx] * len(image_keys))

            # Extract the images
            if search_modality in ["in-domain", "in-domain-rag"] or (
                search_modality == "random" and self._format in ["pt", "dynamic_pt"]
            ):
                all_retrieved_images = {
                    v: _in_domain_images[v].convert("RGB") for v in set(all_images_keys)
                }
            else:
                all_retrieved_images = self.get_images(all_images_keys)

            if rag.get("resize_images", None) is not None:
                for key in all_retrieved_images:
                    image = all_retrieved_images[key]
                    if image is not None and max(image.size) > rag["resize_images"]:
                        all_retrieved_images[key].thumbnail(
                            (rag["resize_images"], rag["resize_images"]), Image.LANCZOS
                        )

            # Format the retrieved data for each sample
            rag_data_set = []
            for _, results in enumerate(results_set):
                if limit is not None:
                    results = results[:limit]

                image_keys = [x.get("image_path") for x in results]

                if prepare is None:
                    raise ValueError(
                        "When retrieving images, a `prepare` function must be provided to convert"
                        " images to the desired format."
                    )

                retrieved_images = [
                    prepare(all_retrieved_images.get(key))
                    for key in image_keys
                    if key in all_retrieved_images
                ]

                # Text formatting
                if format == "list-captions":
                    captions = [
                        f"{idx + 1}. {result.get(caption_key)}"
                        for idx, result in enumerate(results)
                    ]
                    rag_text = "\n".join(captions)
                    rag_data = self._prepare_context_with_images(
                        prompts, retrieved_images, rag_text, format
                    )

                elif format == "cased":
                    captions = [result.get(caption_key) for result in results]
                    vocabularies = self._vocab_transform(captions)
                    words = list(set([vocab or ["object"] for vocab in vocabularies]))

                    rag_text = ", ".join(words)
                    rag_data = self._prepare_context_with_images(
                        prompts, retrieved_images, rag_text, format
                    )

                elif format == "no-text":
                    rag_text = ""
                    rag_data = self._prepare_context_with_images(
                        prompts, retrieved_images, rag_text, format
                    )

                elif format == "interleaved":
                    rag_data = [
                        {"type": "text", "text": prompts[0]},
                    ]
                    for image, result in zip(retrieved_images, results, strict=True):
                        rag_data.append(image)
                        rag_data.append(
                            {"type": "text", "text": prompts[-1].format(result.get(caption_key))}
                        )

                if result_callback is not None:
                    rag_data = result_callback(rag_data, images=retrieved_images)  # type: ignore

                rag_data_set.append(rag_data)

            return rag_data_set

        # =========================== #
        # Use the retrieved text data #
        # =========================== #
        elif grab_data_type == "text":
            format = rag.get("format", "list-captions")
            prompt = rag.get("prompt", "{}")
            caption_key = rag.get("caption_key", "caption")
            limit = rag.get("limit", None)

            for i in range(len(results_set)):
                if limit is not None:
                    results_set[i] = results_set[i][:limit]

            if format == "list-captions":
                rag_data_set = []
                for results in results_set:
                    rag_data = [
                        f"{idx + 1}. {result.get(caption_key, '').strip()}"
                        for idx, result in enumerate(results)
                    ]
                    rag_data = "\n".join(rag_data)
                    rag_data_set.append(rag_data)

            elif format == "bullet-list-captions":
                rag_data_set = []
                for results in results_set:
                    captions = ["- " + result.get(caption_key, "").strip() for result in results]
                    rag_data = "\n".join(captions)
                    rag_data_set.append(rag_data)

            elif format == "newlines":
                rag_data_set = []
                for results in results_set:
                    captions = [result.get(caption_key, "").strip() for result in results]
                    rag_data = "\n".join(captions)
                    rag_data_set.append(rag_data)

            elif format == "csv":
                rag_data_set = []
                for results in results_set:
                    captions = [result.get(caption_key, "").strip() for result in results]
                    rag_data = ",".join(captions)
                    rag_data_set.append(rag_data)

            elif format == "csv-spaces":
                rag_data_set = []
                for results in results_set:
                    captions = [result.get(caption_key, "").strip() for result in results]
                    rag_data = ", ".join(captions)
                    rag_data_set.append(rag_data)

            elif format == "cased":
                rag_data_set = []
                for results in results_set:
                    captions = [result.get(caption_key) for result in results]
                    vocabularies = self._vocab_transform(captions)
                    words = list(set([vocab or ["object"] for vocab in vocabularies]))

                    # Shuffle the words
                    if rag.get("shuffle", False):
                        random.shuffle(words)

                    rag_data = ", ".join(words)
                    rag_data_set.append(rag_data)

            else:
                raise NotImplementedError(f"RAG format '{format}' is not supported.")

            # Format the prompt with the retrieved data for each sample
            prompts = [prompt.format(rag_data) for rag_data in rag_data_set]

            # Return a dictionary for each input sample
            # The dictionary is inside a list to be compatible with the image format
            rag_data_set = [
                [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
                for prompt in prompts
            ]

            if result_callback is not None:
                rag_data_set = [
                    result_callback(rag_data, images=None) for rag_data in rag_data_set
                ]

            return rag_data_set

        else:
            raise NotImplementedError(
                f"Search modality '{search_modality}' is not supported with grab data type set to '{grab_data_type}'."  # noqa: E501
            )

    def __call__(self, *args, **kwds) -> list[dict]:
        """Alias for the `retrieve` method."""
        return self.retrieve(*args, **kwds)

    def get_images(self, keys: list[str], from_tar: bool = False) -> dict[str, Image.Image | None]:
        """Get images by their keys.

        Args:
        ----
            keys (list[str]): List of image keys.
            from_tar (bool): Whether to load images from tar files. Default is False.

        """
        root = Path(self._path).parent.parent / "images"

        return get_images_by_key(keys, root, from_tar=from_tar)
