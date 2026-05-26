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
        LocalEmbed["MiniLM-L6 (CPU)"]
        
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
    participant Embed as MiniLM-L6 (Local CPU)
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
    
    opt High Precision Cloud Upgrade
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
    
    Router -->|Confidence > 0.85| DirectSearch["Standard Search"]
    Router -->|Low Confidence| Decomp["Query Decomposition"]
    
    Decomp --> SubQ1["Sub-Query 1"]
    Decomp --> SubQ2["Sub-Query 2"]
    
    DirectSearch --> Hybrid["Hybrid Retrieval<br>(BM25 + Dense)"]
    SubQ1 --> Hybrid
    SubQ2 --> Hybrid
    
    Hybrid --> RRF["Reciprocal Rank Fusion (RRF)<br>Merge Results"]
    RRF --> FlashRankRerank["FlashRank<br>(Local CPU)"]
    FlashRankRerank --> Threshold{"Relevance > 0.8?"}
    
    Threshold -->|Yes| Final["Final Context Payload"]
    Threshold -->|No| Fallback["Return Raw Evidence or Clarify"]
```

---

### 5. Agent Communication Flow
*The LangGraph cyclic state machine. (Invoked only on low-confidence route).*

```mermaid
graph TD
    Supervisor{"Supervisor Agent"}
    
    ResearchAgent["Research Sub-Agent<br>(Vector Search)"]
    Critic["Evaluator / Critic"]
    
    Supervisor -->|Routes to| ResearchAgent
    
    ResearchAgent -->|Returns data| Supervisor
    
    Supervisor -->|Requests Review| Critic
    Critic -->|Passes| End["Final Generation"]
    Critic -->|Fails: 'Needs more info'<br>(Max 2 loops)| Supervisor
    
    classDef agent fill:#44337a,stroke:#b794f4,color:#faf5ff
    class Supervisor,ResearchAgent,Critic agent
```

---

### 6. Storage Architecture Map
*How data is physically stored locally vs transiently.*

```mermaid
graph LR
    subgraph "Local Disk (Persistent)"
        VDB[("sqlite-vec<br>(Embeddings & Metadata)")]
        SQL[("SQLite<br>(Raw Doc Markdown)")]
        PG[("SQLite Checkpointer<br>(LangGraph Checkpoints)")]
    end
    
    subgraph "System RAM (16GB) - Transient"
        InMemCache["In-process cache (dict)"]
        ActiveGraph["Active LangGraph State"]
        EmbModel["Embedder Weights<br>(~0.5 GB CPU)"]
    end

    ActiveGraph -->|Reads/Writes| PG
    ActiveGraph -->|Reads| SQL
    ActiveGraph -->|Searches| VDB
    ActiveGraph -->|Checks| InMemCache
    ActiveGraph -->|Passes strings to| EmbModel
```

---

### 7. Ingestion Pipeline Map
*How complex files are processed before hitting the database.*

```mermaid
graph TD
    Input["Raw Files<br>(PDF, DOCX, Code)"]
    
    Input --> PDFType{"Is PDF text-extractable?"}
    
    PDFType -->|Yes (Text)| PyMuPDF["PyMuPDF<br>(Local CPU Parsing)"]
    PDFType -->|No (Scanned/Complex)| LlamaParse{"LlamaParse API<br>(Cloud VLM Extraction)"}
    
    PyMuPDF --> Cleaner["Local Text Cleaner<br>(Regex / Normalization)"]
    LlamaParse --> Cleaner
    
    Cleaner --> SemanticChunking["Semantic Markdown Splitter"]
    
    SemanticChunking --> ChunkText["Context-Aware Chunks"]
    
    ChunkText --> MiniLMEmbed["MiniLM-L6 Embedding Model<br>(CPU)"]
    
    MiniLMEmbed -->|Dense Vectors| VecDB[("sqlite-vec")]
    ChunkText -->|Raw Text| SQL[("Document Store")]
    
    classDef pipe fill:#1a202c,stroke:#a0aec0,color:#f7fafc
    class Input,Cleaner,SemanticChunking,ChunkText,MiniLMEmbed pipe
```
