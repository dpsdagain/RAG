"""Embedding model wrapper using ONNX Runtime for bge-small-en-v1.5.

Provides synchronous embed() and embed_batch() methods optimized for
CPU-first execution with configurable thread counts and L2 normalization.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.infrastructure.observability import get_logger, metrics

logger = get_logger("embedder")


class Embedder:
    """ONNX Runtime wrapper for sentence embedding models.

    Optimized for CPU inference with controllable thread counts.
    All embeddings are L2-normalized before returning.
    """

    def __init__(
        self,
        model_path: str,
        dim: int = 384,
        num_threads: int = 4,
    ) -> None:
        """Initialize the embedder.

        Args:
            model_path: HuggingFace model ID or local path to ONNX model.
            dim: Expected embedding dimension.
            num_threads: Number of CPU threads for ONNX Runtime.

        Raises:
            FileNotFoundError: If model files cannot be found or downloaded.
        """
        self._dim = dim
        self._num_threads = num_threads
        self._model_path = model_path
        self._session: Any = None
        self._tokenizer: Any = None

        # Set thread limits before importing ONNX
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model and tokenizer."""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Install with: pip install onnxruntime tokenizers"
            ) from e

        # Resolve model path
        model_dir = self._resolve_model_path(self._model_path)
        onnx_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                f"Run: python -c \"from app.models.embedder import download_model; "
                f"download_model('{self._model_path}')\" to download it."
            )

        # Configure ONNX session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        if tokenizer_path.exists():
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            # Fallback: try loading from HuggingFace tokenizers
            try:
                self._tokenizer = Tokenizer.from_pretrained(self._model_path)
            except Exception:
                raise FileNotFoundError(
                    f"Tokenizer not found at {tokenizer_path} and could not "
                    f"download from {self._model_path}."
                )

        logger.info(
            "embedder_loaded",
            model=self._model_path,
            dim=self._dim,
            threads=self._num_threads,
        )

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        """Resolve a model path (local directory or HuggingFace cache)."""
        # Check if it's already a local directory
        local_path = Path(model_path)
        if local_path.is_dir() and (local_path / "model.onnx").exists():
            return local_path

        # Check HuggingFace cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Try common cache patterns
        model_slug = model_path.replace("/", "--")
        candidates = [
            cache_dir / f"models--{model_slug}" / "snapshots",
            Path.home() / ".cache" / "rag" / "models" / model_slug,
        ]
        for candidate in candidates:
            if candidate.exists():
                # Get latest snapshot
                snapshots = sorted(candidate.iterdir(), reverse=True)
                for snap in snapshots:
                    if (snap / "model.onnx").exists():
                        return snap

        # Return default cache location for download_model()
        default_dir = Path.home() / ".cache" / "rag" / "models" / model_slug
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.
            input_type: Accepted for interface parity with cloud embedders
                        (e.g. Voyage query/document prompts). Ignored here —
                        bge-small has no input-type conditioning.

        Returns:
            L2-normalized embedding vector of length self._dim.
        """
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(
        self, texts: list[str], batch_size: int = 64, input_type: str = "document"
    ) -> list[list[float]]:
        """Embed multiple texts in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per inference batch.
            input_type: Ignored (interface parity — see embed()).

        Returns:
            List of L2-normalized embedding vectors.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            with metrics.timer("embed_batch"):
                # Tokenize
                encodings = self._tokenizer.encode_batch(batch)

                # Build input arrays
                max_len = min(max(len(e.ids) for e in encodings), 512)  # Cap at 512 tokens
                input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
                attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
                token_type_ids = np.zeros((len(batch), max_len), dtype=np.int64)

                for j, encoding in enumerate(encodings):
                    length = min(len(encoding.ids), max_len)
                    input_ids[j, :length] = encoding.ids[:length]
                    attention_mask[j, :length] = encoding.attention_mask[:length]
                    if encoding.type_ids:
                        token_type_ids[j, :length] = encoding.type_ids[:length]

                # Run inference
                feeds: dict[str, np.ndarray] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                # Some models expect token_type_ids
                input_names = [inp.name for inp in self._session.get_inputs()]
                if "token_type_ids" in input_names:
                    feeds["token_type_ids"] = token_type_ids

                outputs = self._session.run(None, feeds)

                # Mean pooling over token embeddings (masked)
                token_embeddings = outputs[0]  # (batch, seq_len, dim)
                mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
                sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
                sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                mean_embeddings = sum_embeddings / sum_mask

                # L2 normalize
                normalized = self._normalize(mean_embeddings)

                for vec in normalized:
                    all_embeddings.append(vec[:self._dim].tolist())

        return all_embeddings

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 normalize a batch of vectors.

        Args:
            vectors: (N, dim) array of vectors.

        Returns:
            L2-normalized vectors.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vectors / norms

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        encoding = self._tokenizer.encode(text)
        return len(encoding.ids)

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self._dim


class VoyageEmbedder:
    """Cloud embedder backed by Voyage AI (voyage-code-3 by default).

    Drop-in replacement for ``Embedder`` — same embed/embed_batch/
    count_tokens/dim surface — but calls the Voyage API instead of running
    a local ONNX model. Chosen for a codebase+document corpus: voyage-code-3
    is tuned for code retrieval *and* documentation.

    Notes:
      * input_type matters for retrieval quality: documents are embedded with
        "document" and queries with "query" (Voyage prepends a tuned prompt).
      * Outputs are L2-normalized so cosine ranking via sqlite-vec L2 distance
        is preserved, matching the local Embedder's contract.
      * count_tokens uses Voyage's own tokenizer (local after a one-time
        fetch) with a char-based fallback so chunking never hits the network
        per sentence in a degraded environment.
    """

    _MAX_BATCH = 128          # Voyage per-request text cap
    _VALID_INPUT = ("query", "document")

    def __init__(
        self,
        model: str = "voyage-code-3",
        dim: int = 1024,
        api_key: str = "",
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._dim = dim
        self._max_retries = max(1, max_retries)
        self._tokenizer_failed = False

        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "voyageai is required for the Voyage embedder. "
                "Install it with: pip install voyageai"
            ) from e

        key = (
            api_key
            or os.getenv("RAG_EMBEDDING__API_KEY", "")
            or os.getenv("VOYAGE_API_KEY", "")
        )
        if not key:
            raise ValueError(
                "Voyage API key missing. Set RAG_EMBEDDING__API_KEY in .env "
                "(or VOYAGE_API_KEY) to use the Voyage embedder."
            )

        self._client = voyageai.Client(api_key=key)
        logger.info("embedder_loaded", model=model, dim=dim, backend="voyage")

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        """Embed a single text. ``input_type`` is 'query' or 'document'."""
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(
        self, texts: list[str], batch_size: int = 128, input_type: str = "document"
    ) -> list[list[float]]:
        """Embed many texts, batching to respect Voyage's per-request cap."""
        if not texts:
            return []
        it = input_type if input_type in self._VALID_INPUT else "document"
        bs = max(1, min(batch_size, self._MAX_BATCH))

        # Voyage rejects empty/whitespace-only strings ("Input cannot contain
        # empty strings"), but code chunkers can legitimately emit blank chunks
        # (e.g. empty imports blocks). Local ONNX tolerates them. Replace with a
        # single space to keep the 1:1 index mapping the caller relies on.
        texts = [t if (t and t.strip()) else " " for t in texts]

        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            out.extend(self._embed_call(texts[i: i + bs], it))
        return out

    def _embed_call(self, batch: list[str], input_type: str) -> list[list[float]]:
        """One Voyage embed request with retry/backoff on transient errors."""
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with metrics.timer("embed_batch"):
                    resp = self._client.embed(
                        batch,
                        model=self._model,
                        input_type=input_type,
                        output_dimension=self._dim,
                    )
                return self._normalize(resp.embeddings)
            except Exception as e:  # network / rate-limit / API error
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
        metrics.increment("embed_errors")
        raise RuntimeError(
            f"Voyage embed failed after {self._max_retries} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> list[list[float]]:
        """L2-normalize a batch (idempotent if Voyage already returns unit vecs)."""
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return (arr / norms).tolist()

    def count_tokens(self, text: str) -> int:
        """Token count via Voyage's tokenizer; char-heuristic fallback."""
        if not self._tokenizer_failed:
            try:
                return int(self._client.count_tokens([text], model=self._model))
            except Exception as e:
                # Fall back permanently for this process so we don't retry a
                # broken tokenizer load on every sentence during ingest.
                self._tokenizer_failed = True
                logger.warning("voyage_count_tokens_fallback", error=str(e))
        return max(1, len(text) // 4)

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self._dim


def build_embedder(settings: Any) -> Any:
    """Construct the embedder selected by ``settings.embedding.provider``.

    'voyage'/'voyageai' → cloud VoyageEmbedder (needs an API key).
    Anything else       → local ONNX Embedder (bge-small).
    """
    emb = settings.embedding
    provider = str(getattr(emb, "provider", "onnx")).lower()

    if provider in ("voyage", "voyageai"):
        api_key = (
            getattr(emb, "api_key", "")
            or os.getenv("RAG_EMBEDDING__API_KEY", "")
            or os.getenv("VOYAGE_API_KEY", "")
        )
        return VoyageEmbedder(model=emb.model_path, dim=emb.dim, api_key=api_key)

    return Embedder(
        model_path=emb.model_path,
        dim=emb.dim,
        num_threads=emb.onnx_num_threads,
    )


def download_model(model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> Path:
    """Download ONNX model files from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID.
        cache_dir: Local cache directory. Defaults to ~/.cache/rag/models/.

    Returns:
        Path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    if cache_dir is None:
        model_slug = model_name.replace("/", "--")
        cache_dir = str(Path.home() / ".cache" / "rag" / "models" / model_slug)

    path = snapshot_download(
        repo_id=model_name,
        local_dir=cache_dir,
        allow_patterns=["*.onnx", "*.json", "*.txt"],
    )
    logger.info("model_downloaded", model=model_name, path=path)
    return Path(path)
