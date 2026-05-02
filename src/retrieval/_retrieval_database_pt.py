import copy
import json
import random
from pathlib import Path

import torch


class RetrievalTensorDatabase:
    """Retrieval database for tensors stored in .pt files.

    Args:
    ----
        database_dir (str): Path to the tensor directory.
        device (str): Device to load the tensors on.

    """

    def __init__(
        self,
        database_dir: str,
        device: str = "cpu",
    ) -> None:
        self._database_dir = database_dir
        self._device = device

        path = Path(database_dir).parent
        name = Path(database_dir).name
        embeddings_fp = path / f"{name}.pt"
        metadata = path / f"{name}_metadata.json"

        self._embeddings = torch.load(embeddings_fp, weights_only=True).to(self._device)
        self._embeddings = self._embeddings.t()  # Transpose for easier matmul
        with open(metadata) as f:
            self._metadata = json.load(f)
            self._original_metadata = copy.deepcopy(self._metadata)

    def _group_by_target(self) -> dict:
        """Group the database entries by their target labels."""
        grouped = {}
        for key, meta in self._original_metadata.items():
            target = meta.get("target")
            if target not in grouped:
                grouped[target] = []
            grouped[target].append(key)
        return grouped

    def few_shot(self, num_shots: int) -> None:
        """Restrict the database to few-shot examples.

        Args:
        ----
            num_shots (int): Number of shots per class to keep.

        """
        grouped = self._group_by_target()

        few_shot_examples = {}
        all_selected_keys = []
        new_idx = 0
        for _, keys in grouped.items():
            selected_keys = random.sample(keys, min(num_shots, len(keys)))
            all_selected_keys.extend([int(x) for x in selected_keys])
            for key in selected_keys:
                few_shot_examples[str(new_idx)] = self._original_metadata[key]
                new_idx += 1

        self._metadata = few_shot_examples

        self._embeddings = self._embeddings.T[all_selected_keys].t()

    @property
    def _metadata_provider(self) -> str:
        """Return the metadata provider."""
        return self._metadata

    def _find_closest(
        self, query: torch.Tensor, k: int = 10, batch_size: int = 2048
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the k closest samples in the database to the query tensor.

        Args:
        ----
            query (torch.Tensor): Query tensor of shape (num_queries, embedding_dim).
            k (int): Number of closest samples to return.
            batch_size (int): Batch size for processing the database.

        """
        query = query.to(self._device)
        N = self._embeddings.shape[1]

        similarities = []
        for i in range(0, N, batch_size):
            batch = self._embeddings[:, i : i + batch_size]
            # (num_queries, emb_dim) x (emb_dim, num_batch) -> (num_queries, num_batch)
            sim = torch.matmul(query, batch)
            similarities.append(sim.cpu())

        similarities = torch.cat(similarities, dim=1)  # (num_queries, N)
        top_values, top_indices = torch.topk(similarities, k=k, dim=1)  # (num_queries, k)

        return top_values, top_indices

    def query(
        self,
        query: torch.Tensor,
        num_samples: int = 10,
        batch_size: int = 2048,
        **kwargs,
    ) -> list[list[dict]]:
        """Query the database with a tensor and return the closest samples.

        Args:
        ----
            query (torch.Tensor): Query tensor of shape (num_queries, embedding_dim).
            num_samples (int): Number of closest samples to return for each query.
            batch_size (int): Batch size for processing the database.
            **kwargs: Additional keyword arguments (not used).

        """
        distances, indices = self._find_closest(query, k=num_samples, batch_size=batch_size)

        results = []
        for dists, inds in zip(distances, indices, strict=True):
            res = []
            for dist, ind in zip(dists, inds, strict=True):
                meta = copy.deepcopy(self._metadata[str(ind.item())])
                meta["distance"] = dist.item()
                res.append(meta)
            results.append(res)

        return results
