import copy

import torch
from PIL import Image


class DynamicRetrievalTensorDatabase:
    """Dynamic retrieval database for tensors stored in .pt files.

    Args:
    ----
        device (str): Device to store the tensors on.

    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self._device = device

        self._embeddings = None
        self._metadata = dict()

    @property
    def _metadata_provider(self) -> dict:
        """Return the metadata provider."""
        return self._metadata

    def _find_closest(
        self, query: torch.Tensor, k: int = 10, batch_size: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the k closest samples in the database to the query tensor.

        Args:
        ----
            query (torch.Tensor): Query tensor of shape (num_queries, embedding_dim).
            k (int): Number of closest samples to return.
            batch_size (int): Batch size for processing the database.

        """
        assert self._embeddings is not None, "Database embeddings are not set."

        query = query.to(self._device)
        N = self._embeddings.shape[0]

        similarities = []
        for i in range(0, N, batch_size):
            batch = self._embeddings[i : i + batch_size]
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
        batch_size: int = 1024,
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
        num_samples = min(num_samples, len(self._metadata))
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

    def add_data(
        self,
        embeddings: torch.Tensor,
        metadata: list[tuple[Image.Image, str]],
    ) -> None:
        """Add data to the database.

        Args:
        ----
            embeddings (torch.Tensor): Embeddings tensor of shape (N, embedding_dim).
            metadata (list[str]): List of labels for each embedding.

        """
        assert embeddings.shape[0] == len(metadata), "Embeddings and metadata size mismatch."

        metadata = [
            {
                "visual": meta[0],
                "target": meta[1],
            }
            for meta in metadata
        ]

        if self._embeddings is None:
            self._embeddings = embeddings.to(self._device).t()  # transpose for search
            self._metadata = {str(i): meta for i, meta in enumerate(metadata)}

        else:
            self._embeddings = torch.cat(
                [self._embeddings.T, embeddings.to(self._device)], dim=0
            ).t()  # transpose for search
            current_size = len(self._metadata)
            for i, meta in enumerate(metadata):
                self._metadata[str(current_size + i)] = meta
