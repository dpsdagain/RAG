"""Stack A/B benchmark on a code (or any file) corpus.

Compares two retrieval stacks end-to-end on the SAME corpus:
  * OLD = local bge-small (dim 384) + FlashRank ms-marco
  * NEW = Voyage voyage-code-3 (dim 1024) + rerank-2.5

For each stack it ingests the corpus via the real ingestion path (tree-sitter
+ CodeChunker for code, SemanticChunker otherwise), then runs a golden set of
queries through hybrid-search -> rerank -> real LLM and scores:
  * retrieval@5 — any `must_retrieve` substring in the top-5 reranked chunks
  * generation  — any `must_generate` substring in the real LLM answer

This is the one eval the pytest `tests/quality/` suite does NOT cover: a
side-by-side comparison of two different embedder/reranker stacks. Use it to
answer "did my embedder/reranker/pipeline change actually help?".

Usage:
    .venv/Scripts/python.exe scripts/benchmark_code.py [--code-dir DIR] [--golden FILE.json]

  --code-dir  Folder to ingest (recursively). Default: the bundled clean_Code path.
  --golden    JSON file: a list of {id, query, must_retrieve[], must_generate[]}.
              Substrings are matched case-insensitively. Defaults to the built-in
              set below (authored for clean_Code — replace it for other corpora).

Needs RAG_EMBEDDING__API_KEY (Voyage) and RAG_LLM__API_KEY in .env.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RAG_SYSTEM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_SYSTEM))

from app.infrastructure.config import get_settings           # noqa: E402
from app.infrastructure.database import Database, DocumentRecord  # noqa: E402
from app.ingestion.router import IngestionRouter             # noqa: E402
from app.ingestion.chunking.code_chunker import CodeChunker  # noqa: E402
from app.ingestion.chunking.semantic_chunker import SemanticChunker  # noqa: E402
from app.models.embedder import build_embedder               # noqa: E402
from app.models.llm_client import LLMClient                  # noqa: E402
from app.core.pipeline import RAGPipeline                    # noqa: E402

SCORE_K = 5
DEFAULT_CODE_DIR = Path(r"F:\Gemini_anti\new_llm_v3\new_llm_v2_modified\clean_Code")

# Built-in golden set — authored for the clean_Code corpus. For other corpora,
# pass --golden pointing at your own JSON file with the same shape.
DEFAULT_GOLDEN = [
    {"id": "hybrid_weights", "query": "What weights does hybrid search use for BM25 versus vector search?",
     "must_retrieve": ["bm25_weight", "0.65"], "must_generate": ["0.65", "0.35"]},
    {"id": "rerank_model", "query": "Which cross-encoder reranker model is configured and how many results does it keep?",
     "must_retrieve": ["rerank_model", "ms-marco"], "must_generate": ["ms-marco", "8"]},
    {"id": "cache_threshold", "query": "What cosine similarity threshold does the semantic cache use for a hit?",
     "must_retrieve": ["semantic_cache_threshold", "0.98"], "must_generate": ["0.98"]},
    {"id": "chunk_sizes", "query": "What are the chunk sizes for general text versus code?",
     "must_retrieve": ["chunk_size", "code_chunk_size", "1500", "1000"], "must_generate": ["1500", "1000"]},
    {"id": "embedding_model", "query": "Which embedding model does this application use?",
     "must_retrieve": ["bge-small", "embedding_model_name"], "must_generate": ["bge-small"]},
    {"id": "specialist_code", "query": "Which model is auto-selected for CODE specialist queries?",
     "must_retrieve": ["specialist_mapping", "qwen2.5"], "must_generate": ["qwen2.5", "qwen"]},
    {"id": "ast_chunker", "query": "How does the system chunk source code files using the AST?",
     "must_retrieve": ["codeastchunker", "ast"], "must_generate": ["ast", "chunk"]},
    {"id": "build_rag_chain", "query": "What does the build_rag_chain function set up?",
     "must_retrieve": ["build_rag_chain"], "must_generate": ["retriev", "chain", "rag"]},
    {"id": "vector_router", "query": "What is the VectorRouter class responsible for?",
     "must_retrieve": ["vectorrouter", "class vectorrouter"], "must_generate": ["rout", "vector"]},
    {"id": "bm25_impl", "query": "How is BM25 keyword search implemented in the backend?",
     "must_retrieve": ["sqlitefts5bm25", "fts5"], "must_generate": ["fts5", "bm25"]},
    {"id": "compress_history", "query": "How does the system compress long chat history?",
     "must_retrieve": ["compress_chat_history"], "must_generate": ["history", "compress", "token"]},
    {"id": "cosine_fn", "query": "Is there a helper to compute cosine similarity between two vectors?",
     "must_retrieve": ["calculate_cosine_similarity", "cosine"], "must_generate": ["cosine", "similarit"]},
    {"id": "force_retrieval", "query": "How does the app decide to force retrieval for a query?",
     "must_retrieve": ["detect_force_retrieval", "force"], "must_generate": ["force", "retriev"]},
    {"id": "ingest_chroma", "query": "What does the ingest_into_chroma function do?",
     "must_retrieve": ["ingest_into_chroma", "chroma"], "must_generate": ["chroma", "ingest"]},
]


def hit(chunks_or_text, substrings, is_text=False) -> bool:
    if is_text:
        low = chunks_or_text.lower()
        return any(sub in low for sub in substrings)
    top = [c.content.lower() for c in chunks_or_text[:SCORE_K]]
    return any(any(sub in content for sub in substrings) for content in top)


def make_config(base, provider, model_path, dim, rerank_model):
    cfg = base.model_copy(deep=True)
    cfg.embedding.provider = provider
    cfg.embedding.model_path = model_path
    cfg.embedding.dim = dim
    cfg.retrieval.rerank_model = rerank_model
    # Isolate the embedder+reranker effect; keep both stacks identical otherwise.
    cfg.pipeline.crag_enabled = False
    cfg.pipeline.decomposition_enabled = False
    cfg.pipeline.faithfulness_check_enabled = False
    cfg.pipeline.agentic_enabled = False
    cfg.cache.similarity_threshold = 2.0  # >1.0 -> cache never hits
    return cfg


async def ingest_folder(db, embedder, code_dir: Path) -> int:
    router = IngestionRouter()
    code_chunker = CodeChunker(embedder)
    sem_chunker = SemanticChunker(embedder)
    total = 0
    for i, path in enumerate(sorted(code_dir.rglob("*"))):
        if not path.is_file():
            continue
        try:
            parsed = await router.parse_file(path)
        except Exception as e:
            print(f"    [skip {path.name}: {e}]")
            continue
        doc_id = f"doc_{i}_{path.stem}"
        await db.insert_document(DocumentRecord(
            doc_id=doc_id, source_uri=str(path),
            source_type=parsed.source_type or "code",
            file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            title=path.name, parser_used="treesitter", status="active",
        ))
        if parsed.source_type == "code":
            chunks = code_chunker.chunk(parsed, doc_id, contextual_header="")
        else:
            chunks = sem_chunker.chunk(parsed, doc_id)
        await db.insert_chunks_batch(chunks)
        total += len(chunks)
        print(f"    {path.name}: {len(chunks)} chunks ({parsed.source_type})")
    return total


async def run_stack(label, cfg, llm, code_dir, golden, tmp_dir) -> dict:
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    embedder = build_embedder(cfg)
    db = Database(tmp_dir / f"{label.split()[0]}.db", dim=cfg.embedding.dim)
    await db.initialize()
    n_chunks = await ingest_folder(db, embedder, code_dir)
    print(f"  ingested {n_chunks} chunks total (dim={cfg.embedding.dim})")

    pipe = RAGPipeline(db=db, embedder=embedder, llm=llm, settings=cfg)
    r_pass, g_pass, lat, per_q = 0, 0, [], {}
    for case in golden:
        hits = await pipe.search_engine.search(case["query"], top_n=10)
        reranked = await pipe.reranker.rerank(case["query"], hits, top_k=15)
        r_ok = hit(reranked, [s.lower() for s in case["must_retrieve"]])
        t0 = time.perf_counter()
        result = await pipe.execute(query=case["query"])
        lat.append((time.perf_counter() - t0) * 1000)
        g_ok = hit(result.response, [s.lower() for s in case["must_generate"]], is_text=True)
        r_pass += r_ok
        g_pass += g_ok
        per_q[case["id"]] = (r_ok, g_ok)
        print(f"  {'R+' if r_ok else 'R-'} {'G+' if g_ok else 'G-'}  {case['id']}")
    db.close()
    n = len(golden)
    print(f"  --> retrieval@5: {r_pass}/{n} ({100*r_pass/n:.0f}%)  |  "
          f"generation: {g_pass}/{n} ({100*g_pass/n:.0f}%)  |  avg {sum(lat)/len(lat):.0f} ms")
    return {"label": label, "n": n, "r": r_pass, "g": g_pass, "ms": sum(lat)/len(lat), "per_q": per_q}


async def main(code_dir: Path, golden: list[dict]) -> None:
    if not code_dir.exists():
        print(f"FATAL: corpus dir not found: {code_dir}")
        return
    base = get_settings()
    llm = LLMClient(provider=base.llm.provider, base_url=base.llm.base_url,
                    model=base.llm.model, api_key=base.llm.api_key,
                    timeout=base.llm.timeout_seconds)
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_bench_"))

    old_cfg = make_config(base, "onnx", "BAAI/bge-small-en-v1.5", 384, "ms-marco-MiniLM-L-12-v2")
    new_cfg = make_config(base, "voyage", "voyage-code-3", 1024, "rerank-2.5")

    results = []
    try:
        for label, cfg in (("OLD bge-small+FlashRank", old_cfg),
                           ("NEW voyage-code-3+rerank-2.5", new_cfg)):
            try:
                results.append(await run_stack(label, cfg, llm, code_dir, golden, tmp_dir))
            except Exception as e:
                import traceback
                print(f"\n[{label} FAILED: {type(e).__name__}: {e}]")
                traceback.print_exc()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(results) == 2:
        old, new = results
        n = old["n"]
        print(f"\n{'='*72}\n  COMPARISON  (n={n} queries, corpus={code_dir.name})\n{'='*72}")
        print(f"  retrieval@5    OLD {old['r']}/{n} ({100*old['r']/n:.0f}%)    "
              f"NEW {new['r']}/{n} ({100*new['r']/n:.0f}%)    delta {new['r']-old['r']:+d}")
        print(f"  generation     OLD {old['g']}/{n} ({100*old['g']/n:.0f}%)    "
              f"NEW {new['g']}/{n} ({100*new['g']/n:.0f}%)    delta {new['g']-old['g']:+d}")
        print(f"  avg latency    OLD {old['ms']:.0f} ms    NEW {new['ms']:.0f} ms")
        ri = [q for q in old["per_q"] if not old["per_q"][q][0] and new["per_q"][q][0]]
        rr = [q for q in old["per_q"] if old["per_q"][q][0] and not new["per_q"][q][0]]
        gi = [q for q in old["per_q"] if not old["per_q"][q][1] and new["per_q"][q][1]]
        gr = [q for q in old["per_q"] if old["per_q"][q][1] and not new["per_q"][q][1]]
        print(f"  retrieval: NEW fixed {ri or '-'} | NEW broke {rr or '-'}")
        print(f"  generation: NEW fixed {gi or '-'} | NEW broke {gr or '-'}")


def _parse_args():
    p = argparse.ArgumentParser(description="Stack A/B benchmark on a corpus.")
    p.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR,
                   help="Folder to ingest recursively (default: bundled clean_Code path).")
    p.add_argument("--golden", type=Path, default=None,
                   help="JSON golden set: [{id, query, must_retrieve[], must_generate[]}].")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    golden_set = DEFAULT_GOLDEN
    if args.golden is not None:
        golden_set = json.loads(args.golden.read_text(encoding="utf-8"))
    asyncio.run(main(args.code_dir, golden_set))
