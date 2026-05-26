"""Ragas evaluation runner for the RAG pipeline."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_evaluation():
    """Run evaluation against the golden set queries."""
    from app.infrastructure.config import get_settings
    from app.infrastructure.database import Database
    from app.models.embedder import Embedder
    from app.models.llm_client import LLMClient
    from app.core.pipeline import RAGPipeline

    settings = get_settings()

    # Load golden set
    golden_dir = Path(__file__).parent.parent / "tests" / "golden_set"
    queries_path = golden_dir / "queries.json"

    if not queries_path.exists():
        print("ERROR: Golden set not found. Run tests first.")
        return

    with open(queries_path) as f:
        queries = json.load(f)

    # Initialize pipeline
    db = Database(settings.database.db_path)
    await db.initialize()
    embedder = Embedder(
        model_path=settings.embedding.model_path,
        dim=settings.embedding.dim,
    )
    llm = LLMClient(
        provider=settings.llm.provider,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        api_key=settings.llm.api_key,
    )
    pipeline = RAGPipeline(db=db, embedder=embedder, llm=llm, settings=settings)

    # Run queries
    results = []
    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {q['query'][:80]}")
        try:
            response = await pipeline.execute(query=q["query"], stream=False)
            results.append({
                "query": q["query"],
                "expected_type": q.get("type", "unknown"),
                "response": response.response[:200],
                "sources": len(response.sources),
                "crag_verdict": response.crag_verdict,
                "latency_ms": response.latency_ms,
            })
            print(f"  → {response.crag_verdict} | {len(response.sources)} sources | {response.latency_ms:.0f}ms")
        except Exception as e:
            print(f"  → ERROR: {e}")
            results.append({"query": q["query"], "error": str(e)})

    # Save results
    output_path = golden_dir / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    successful = [r for r in results if "error" not in r]
    if successful:
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        print(f"  Queries: {len(queries)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Avg Latency: {avg_latency:.0f}ms")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
