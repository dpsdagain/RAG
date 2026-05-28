"""Semantic chunker: embedding-aware document segmentation.

Splits documents into semantically coherent chunks using inter-sentence
cosine similarity, respecting Markdown structure, tables, code blocks,
and token budget constraints.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import numpy as np

from app.infrastructure.database import ChunkRecord
from app.infrastructure.observability import get_logger, metrics
from app.ingestion.chunking.adaptive_metrics import (
    compute_dcc,
    compute_icc,
    evaluate_chunk_quality,
)
from app.ingestion.parsers.base import ParsedDocument
from app.models.embedder import Embedder

logger = get_logger("chunking.semantic_chunker")


class SemanticChunker:
    """Embedding-aware document chunker.

    Algorithm:
    1. Split text into sentences.
    2. Batch-embed all sentences.
    3. Compute inter-sentence cosine similarity.
    4. Split where similarity drops below threshold or structural boundaries.
    5. Merge small chunks, hard-split oversized ones.
    6. Assign parent chunk IDs for hierarchical retrieval.
    """

    def __init__(
        self,
        embedder: Embedder,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50,
        semantic_threshold: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._max_tokens = max_chunk_tokens
        self._min_tokens = min_chunk_tokens
        self._threshold = semantic_threshold

    def chunk(self, parsed_doc: ParsedDocument, doc_id: str) -> list[ChunkRecord]:
        """Chunk a parsed document into semantically coherent segments.

        Args:
            parsed_doc: The parsed document content.
            doc_id: Document ID to associate chunks with.

        Returns:
            List of ChunkRecord objects with embeddings already computed.
        """
        content = parsed_doc.content
        if not content or not content.strip():
            return []

        # Step 1: Extract special blocks (tables, code) and split into sentences
        special_blocks, clean_text = self._extract_special_blocks(content)
        sentences = self._split_sentences(clean_text)

        if not sentences:
            return []

        # Step 2: Batch-embed all sentences
        with metrics.timer("chunk_embedding"):
            sentence_embeddings = self._embedder.embed_batch(
                [s["text"] for s in sentences]
            )

        # Step 3: Compute inter-sentence similarities
        similarities = self._compute_similarities(sentence_embeddings)

        # Step 4: Identify split points
        split_points = self._find_split_points(sentences, similarities)

        # Step 5: Group sentences into raw chunks
        raw_chunks = self._group_chunks(sentences, split_points)

        # Step 6: Add special blocks as their own chunks
        for block in special_blocks:
            raw_chunks.append({
                "text": block["text"],
                "section_title": block.get("section_title"),
                "source_page": block.get("source_page"),
                "is_special": True,
            })

        # Step 7: Post-process (merge small, split large)
        processed = self._post_process(raw_chunks)

        # Compute a document-level embedding — the L2-normalized mean of
        # all sentence embeddings. Used by DCC to score each chunk's
        # contextual coherence against the document as a whole.
        doc_embedding = self._compute_doc_embedding(sentence_embeddings)

        # Step 8: Build ChunkRecords with embeddings + ICC/DCC metrics
        chunk_records = self._build_records(
            processed, doc_id, parsed_doc,
            sentence_embeddings=sentence_embeddings,
            doc_embedding=doc_embedding,
        )

        metrics.record_latency("chunking_count", len(chunk_records))
        logger.info(
            "document_chunked",
            doc_id=doc_id,
            sentences=len(sentences),
            chunks=len(chunk_records),
        )

        return chunk_records

    @staticmethod
    def _compute_doc_embedding(sentence_embeddings: list[list[float]]) -> list[float]:
        """Mean of sentence embeddings, L2-renormalized.

        Cheap proxy for the document's overall theme without doing a
        second pass through the embedder on the whole-document text.
        """
        if not sentence_embeddings:
            return []
        arr = np.array(sentence_embeddings, dtype=np.float32)
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm < 1e-12:
            return mean_vec.tolist()
        return (mean_vec / norm).tolist()

    def _extract_special_blocks(self, text: str) -> tuple[list[dict], str]:
        """Extract tables and code blocks, replacing them with markers."""
        special: list[dict] = []
        clean = text

        # Extract code blocks
        code_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        for match in code_pattern.finditer(text):
            marker = f"__CODE_BLOCK_{len(special)}__"
            special.append({"text": match.group(), "type": "code"})
            clean = clean.replace(match.group(), marker, 1)

        # Extract tables (lines with | pipes)
        table_pattern = re.compile(
            r'(\|.+\|(?:\n\|[-:| ]+\|)?(?:\n\|.+\|)+)', re.MULTILINE
        )
        for match in table_pattern.finditer(clean):
            marker = f"__TABLE_BLOCK_{len(special)}__"
            special.append({"text": match.group(), "type": "table"})
            clean = clean.replace(match.group(), marker, 1)

        return special, clean

    def _split_sentences(self, text: str) -> list[dict]:
        """Split text into sentences, respecting Markdown headers."""
        sentences: list[dict] = []
        current_section: str | None = None
        current_page: int | None = None

        # Split on paragraph boundaries first
        paragraphs = re.split(r'\n{2,}', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check for Markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', para)
            if header_match:
                current_section = header_match.group(2).strip()
                sentences.append({
                    "text": para,
                    "is_header": True,
                    "section_title": current_section,
                    "source_page": current_page,
                })
                continue

            # Split into sentences within the paragraph
            sent_parts = re.split(r'(?<=[.!?])\s+', para)
            for part in sent_parts:
                part = part.strip()
                if part:
                    sentences.append({
                        "text": part,
                        "is_header": False,
                        "section_title": current_section,
                        "source_page": current_page,
                    })

        return sentences

    @staticmethod
    def _compute_similarities(embeddings: list[list[float]]) -> list[float]:
        """Compute cosine similarity between consecutive sentence embeddings."""
        if len(embeddings) < 2:
            return []

        emb_array = np.array(embeddings, dtype=np.float32)
        sims: list[float] = []

        for i in range(len(emb_array) - 1):
            a = emb_array[i]
            b = emb_array[i + 1]
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-12 or norm_b < 1e-12:
                sims.append(0.0)
            else:
                sims.append(float(dot / (norm_a * norm_b)))

        return sims

    def _find_split_points(
        self,
        sentences: list[dict],
        similarities: list[float],
    ) -> list[int]:
        """Identify indices where chunks should be split."""
        split_points: list[int] = []
        accumulated_tokens = 0

        for i, sentence in enumerate(sentences):
            tokens = self._embedder.count_tokens(sentence["text"])
            accumulated_tokens += tokens

            # Split at headers
            if sentence.get("is_header") and i > 0:
                split_points.append(i)
                accumulated_tokens = tokens
                continue

            # Split when similarity drops below threshold
            if i < len(similarities) and similarities[i] < self._threshold:
                split_points.append(i + 1)
                accumulated_tokens = 0
                continue

            # Split when token budget exceeded
            if accumulated_tokens > self._max_tokens:
                split_points.append(i + 1)
                accumulated_tokens = 0

        return sorted(set(split_points))

    def _group_chunks(
        self,
        sentences: list[dict],
        split_points: list[int],
    ) -> list[dict]:
        """Group sentences into chunks based on split points.

        Each returned dict includes ``sentence_indices`` — the original
        positions of its source sentences in the input list. Downstream
        steps use this to look up the sentence embeddings needed for
        ICC computation per chunk.
        """
        chunks: list[dict] = []
        prev = 0

        split_points = [sp for sp in split_points if 0 < sp <= len(sentences)]
        split_points.append(len(sentences))

        for sp in split_points:
            group = sentences[prev:sp]
            if group:
                text = " ".join(s["text"] for s in group)
                section = next(
                    (s["section_title"] for s in group if s.get("section_title")),
                    None,
                )
                page = next(
                    (s["source_page"] for s in group if s.get("source_page") is not None),
                    None,
                )
                chunks.append({
                    "text": text,
                    "section_title": section,
                    "source_page": page,
                    "is_special": False,
                    "sentence_indices": list(range(prev, sp)),
                })
            prev = sp

        return chunks

    def _post_process(self, chunks: list[dict]) -> list[dict]:
        """Merge small chunks and hard-split oversized ones.

        Preserves ``sentence_indices`` so downstream ICC computation can
        still slice the right sentence embeddings:
        - Merged chunks: union of both source-chunk index lists.
        - Hard-split chunks: each split inherits the parent's full index
          list (they share the same source sentences; ICC will be 1.0 by
          definition for word-bounded splits, which is acceptable).
        """
        if not chunks:
            return []

        processed: list[dict] = []
        for chunk in chunks:
            tokens = self._embedder.count_tokens(chunk["text"])

            if tokens < self._min_tokens and processed:
                # Merge with previous chunk
                prev = processed[-1]
                prev["text"] = prev["text"] + "\n\n" + chunk["text"]
                prev["sentence_indices"] = (
                    prev.get("sentence_indices", []) + chunk.get("sentence_indices", [])
                )
            elif tokens > self._max_tokens * 1.5:
                # Hard split at word boundaries — all splits inherit the
                # parent chunk's sentence indices.
                indices = chunk.get("sentence_indices", [])
                words = chunk["text"].split()
                current: list[str] = []
                current_tokens = 0

                for word in words:
                    current.append(word)
                    current_tokens += 1  # Rough estimate
                    if current_tokens >= self._max_tokens:
                        processed.append({
                            "text": " ".join(current),
                            "section_title": chunk.get("section_title"),
                            "source_page": chunk.get("source_page"),
                            "is_special": False,
                            "sentence_indices": indices,
                        })
                        current = []
                        current_tokens = 0

                if current:
                    processed.append({
                        "text": " ".join(current),
                        "section_title": chunk.get("section_title"),
                        "source_page": chunk.get("source_page"),
                        "is_special": False,
                        "sentence_indices": indices,
                    })
            else:
                processed.append(chunk)

        return processed

    def _build_records(
        self,
        chunks: list[dict],
        doc_id: str,
        parsed_doc: ParsedDocument,
        sentence_embeddings: list[list[float]] | None = None,
        doc_embedding: list[float] | None = None,
    ) -> list[ChunkRecord]:
        """Build ChunkRecord objects with embeddings and quality metrics.

        For each chunk we compute:
        - ICC (Intrachunk Cohesion) from the sentence embeddings whose
          indices were recorded for the chunk during grouping.
        - DCC (Document Contextual Coherence) from the chunk's own
          embedding versus the document-level mean embedding.
        Both are stored in ``metadata_json`` along with a quality label.
        """
        if not chunks:
            return []

        # Batch embed all chunk texts
        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed_batch(texts)

        sentence_embeddings = sentence_embeddings or []
        n_sentences = len(sentence_embeddings)
        has_doc_emb = bool(doc_embedding)

        # Build parent chunk map (section → parent_id)
        section_parents: dict[str, str] = {}

        poor_quality_count = 0
        records: list[ChunkRecord] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            content_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()
            token_count = self._embedder.count_tokens(chunk["text"])
            section = chunk.get("section_title")

            # Assign parent chunk ID for section grouping
            parent_id: str | None = None
            if section and section in section_parents:
                parent_id = section_parents[section]
            elif section:
                section_parents[section] = chunk_id

            # --- Quality metrics ---
            # ICC: pull the source-sentence embeddings for this chunk if we
            # have them. Special blocks (tables, code) get ICC=1.0 since
            # they are atomic by design — there's only one "unit" inside.
            indices = chunk.get("sentence_indices") or []
            valid = [j for j in indices if 0 <= j < n_sentences]
            if chunk.get("is_special") or not valid:
                icc_score = 1.0
            else:
                icc_score = compute_icc([sentence_embeddings[j] for j in valid])

            # DCC: cosine between this chunk's embedding and the doc mean.
            dcc_score = compute_dcc(embedding, doc_embedding) if has_doc_emb else 1.0

            quality = evaluate_chunk_quality(icc_score, dcc_score)
            if quality == "poor":
                poor_quality_count += 1

            existing_meta: dict[str, Any] = {}
            metadata_json = json.dumps({
                **existing_meta,
                "icc": round(icc_score, 4),
                "dcc": round(dcc_score, 4),
                "quality": quality,
                "is_special": bool(chunk.get("is_special")),
            })

            records.append(ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=chunk["text"],
                embedding=embedding,
                chunk_index=i,
                parent_chunk_id=parent_id,
                section_title=section,
                source_page=chunk.get("source_page"),
                token_count=token_count,
                content_hash=content_hash,
                metadata_json=metadata_json,
            ))

        if poor_quality_count:
            logger.warning(
                "chunks_with_low_quality",
                doc_id=doc_id,
                poor=poor_quality_count,
                total=len(records),
                msg="Chunks flagged ICC<0.2 — consider re-ingesting with different settings",
            )

        return records
