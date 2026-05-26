# Phase 4 — Final Architecture Maps
## Visual Blueprint for the 2026 Thin-Client RAG System

> [!NOTE]
> Below are the 7 requested architectural diagrams representing the definitive state of your 2026 personalized RAG system (0 VRAM / 16 GB RAM baseline). These diagrams map the "Thin-Client" paradigm, where heavy matrix compute is offloaded to the cloud, and local models are explicitly restricted to CPU execution.

---

### 1. Full System Architecture Diagram
*A high-level view of the entire local-to-cloud split architecture.*

```mermaid
graph TD
    subgraph "Local Machine (16GB RAM / 0 VRAM)"
        UI["User Interface (CLI/Web)"]
        Orchestrator["LangGraph Orchestrator (Python)"]
        APIEmbed["Cohere API (embed-english-v3.0)"]
        
        subgraph "Local Storage Layer"
            VecDB["sqlite-vec (Vector DB)"]
            DocStore["SQLite (Raw Docs)"]
            StateDB["SQLite Checkpointer (Agent State)"]
        end
    end

    subgraph "Cloud Compute & APIs"
        Ollama["Ollama Cloud (LLM Reasoning & Generation)"]
        LlamaParse["LlamaParse API (VLM Parsing)"]
        Cohere["Cohere API (Cross-Encoder Reranking)"]
    end

    UI <--> Orchestrator
    Orchestrator --> LocalEmbed
    LocalEmbed --> VecDB
    Orchestrator <--> VecDB
    Orchestrator <--> DocStore
    Orchestrator <--> StateDB

    Orchestrator <--> Ollama
    Orchestrator <--> LlamaParse
    Orchestrator -.->|Optional Upgrade| Cohere
    
    classDef local fill:#2a4365,stroke:#90cdf4,stroke-width:2px,color:#e2e8f0
    classDef cloud fill:#742a2a,stroke:#feb2b2,stroke-width:2px,color:#e2e8f0
    
    class UI,Orchestrator,LocalEmbed,VecDB,DocStore,StateDB local
    class Ollama,LlamaParse,Cohere cloud
```

---

### 2. Data Flow Diagram (Query Execution)
*How a user query travels from input to final synthesized response.*

```mermaid
sequenceDiagram
    participant User
    participant LangGraph as LangGraph Orchestrator
    participant Embed as Cohere API
    participant VecDB as sqlite-vec (Local)
    participant FlashRank as FlashRank (Local CPU)
    participant Cohere as Cohere Rerank (Cloud, Optional)
    participant Ollama as Ollama Cloud (LLM)

    User->>LangGraph: "What are the core features of project X?"
    LangGraph->>Ollama: [Query Routing/Rewrite Task]
    Ollama-->>LangGraph: Returns Optimized Queries
    
    LangGraph->>Embed: Send Optimized Queries
    Embed-->>LangGraph: Returns Dense Vectors
    
    LangGraph->>VecDB: Retrieve top-K=100 chunks
    VecDB-->>LangGraph: Returns 100 Raw Chunks
    
    LangGraph->>FlashRank: Rerank Top 100
    FlashRank-->>LangGraph: Returns Top-K=15 Chunks
    
    opt If Cohere is DISABLED
        LangGraph->>LangGraph: Truncate to Top-K=5 Chunks
    end
    
    opt High Precision Cloud Upgrade (Cohere ENABLED)
        LangGraph-->>Cohere: Send 15 Chunks + Query for Reranking
        Cohere-->>LangGraph: Returns Top-K=5 Reranked Chunks
    end
    
    LangGraph->>Ollama: Generate response using Top-K=5 Context
    Ollama-->>LangGraph: Streams Final Synthesis
    LangGraph->>User: Streams Final Answer
```

---

### 3. Memory Hierarchy Diagram
*Visualizing the cognitive architecture for temporal and static memory.*

```mermaid
graph BT
    L1["L1: Model Memory<br>(Parametric weights in Cloud LLM)"] 
    L2["L2: Working Memory<br>(LangGraph Context Window & State)"]
    L3["L3: Semantic System Memory<br>(sqlite-vec chunks & documents)"]
    L4["L4: Procedural / Episodic Memory<br>(Versioned preference store - SQLite KV, user-editable)"]

    L4 -->|Injects rules/history| L2
    L3 -->|Injects retrieved facts| L2
    L2 -->|Processed by| L1
    
    classDef mem fill:#234e52,stroke:#81e6d9,color:#e6fffa
    class L1,L2,L3,L4 mem
```

---

### 4. Retrieval Orchestration Diagram
*The logic gate for dynamic retrieval paths.*

```mermaid
graph TD
    Query["User Query"] --> Router{"Confidence Router"}
    
    Router -->|High Confidence| DirectSearch["Standard Search"]
    Router -->|Low Confidence| Decomp["Query Decomposition"]
    
    Decomp --> SubQ1["Sub-Query 1"]
    Decomp --> SubQ2["Sub-Query 2"]
    
    DirectSearch --> Hybrid["Hybrid Retrieval<br>(BM25 + Dense)"]
    SubQ1 --> Hybrid
    SubQ2 --> Hybrid
    
    Hybrid --> RRF["Reciprocal Rank Fusion (RRF)<br>Merge Results"]
    RRF --> FlashRankRerank["FlashRank<br>(Local CPU)"]
    FlashRankRerank --> Evaluator{"Context Evaluator<br>Passes?"}
    
    Evaluator -->|Yes| Final["Final Context Payload"]
    Evaluator -->|No| Fallback["Trigger Rewrite/CRAG"]
```

---

### 5. Linear Pipeline Orchestration
*The predictable linear pipeline replacing the cyclical multi-agent graph.*

```mermaid
graph TD
    Start["User Query"] --> Retrieve["Hybrid Retrieval Phase"]
    
    Retrieve --> Rerank["FlashRank Reranking"]
    
    Rerank --> Evaluate{"Context Evaluator Node"}
    
    Evaluate -->|Context is Good| Generate["Final Generation (Ollama)"]
    
    Evaluate -->|Context Poor| Rewrite["Targeted Query Rewrite<br>(Max 1 Retry)"]
    Rewrite --> Retrieve
    
    Evaluate -->|Context Poor (After Retry)| Fallback["CRAG Web Fallback<br>(Tavily)"]
    Fallback --> Generate
    
    classDef pipe fill:#44337a,stroke:#b794f4,color:#faf5ff
    class Retrieve,Rerank,Evaluate,Rewrite,Fallback pipe
```

---

### 6. Storage Architecture Map
*How data is physically stored locally vs transiently.*

```mermaid
graph LR
    subgraph "Local Disk (Persistent)"
        VDB[("sqlite-vec<br>(Embeddings & Metadata)")]
        SQL[("SQLite<br>(Raw Doc Markdown)")]
        PG[("SQLite Checkpointer<br>(Pipeline State)")]
    end
    
    subgraph "System RAM (16GB) - Transient"
        InMemCache["In-process cache (dict)"]
        ActivePipe["Active Pipeline State"]
    end

    ActivePipe -->|Reads/Writes| PG
    ActivePipe -->|Reads| SQL
    ActivePipe -->|Searches| VDB
    ActivePipe -->|Checks| InMemCache
```

---

### 7. Ingestion Pipeline Map
*How complex files are processed before hitting the database.*

```mermaid
graph TD
    Input["Raw Files<br>(PDF, DOCX, Code)"]
    
    Input --> PDFType{"Is PDF text-extractable?"}
    
    PDFType -->|Yes (Text)| ParserLocal["pymupdf4llm<br>(Local CPU Parsing)"]
    PDFType -->|No (Scanned/Complex)| ParserCloud["LlamaParse API<br>(Cloud VLM Extraction)"]
    
    ParserLocal --> Cleaner["Local Text Cleaner<br>(Regex / Normalization)"]
    ParserCloud --> Cleaner
    
    Cleaner --> SemanticChunking["Semantic Markdown Splitter"]
    
    SemanticChunking --> ChunkText["Context-Aware Chunks"]
    
    ChunkText --> CohereEmbed["Cohere Embedding API<br>(embed-english-v3.0)"]
    
    CohereEmbed -->|Dense Vectors| VecDB[("sqlite-vec")]
    ChunkText -->|Raw Text| SQL[("Document Store")]
    
    classDef pipe fill:#1a202c,stroke:#a0aec0,color:#f7fafc
    class Input,Cleaner,SemanticChunking,ChunkText,CohereEmbed pipe
```
