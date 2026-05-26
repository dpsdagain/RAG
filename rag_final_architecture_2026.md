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
        Orchestrator["Linear Pipeline (Async Python)"]
        LocalEmbed["BAAI/bge-small-en-v1.5 (CPU)"]
        
        subgraph "Local Storage Layer"
            VecDB["sqlite-vec (Vector DB)"]
            DocStore["SQLite (Raw Docs)"]
            StateDB["SQLite (Pipeline Checkpointer)"]
        end
    end

    subgraph "Cloud Compute & APIs"
        Ollama["Ollama Cloud (LLM Reasoning & Generation)"]
        LlamaParse["LlamaParse API (VLM Parsing)"]
    end

    UI <--> Orchestrator
    Orchestrator --> LocalEmbed
    LocalEmbed --> VecDB
    Orchestrator <--> VecDB
    Orchestrator <--> DocStore
    Orchestrator <--> StateDB

    Orchestrator <--> Ollama
    Orchestrator <--> LlamaParse
    
    classDef local fill:#2a4365,stroke:#90cdf4,stroke-width:2px,color:#e2e8f0
    classDef cloud fill:#742a2a,stroke:#feb2b2,stroke-width:2px,color:#e2e8f0
    
    class UI,Orchestrator,LocalEmbed,VecDB,DocStore,StateDB local
    class Ollama,LlamaParse cloud
```

---

### 2. Data Flow Diagram (Query Execution)
*How a user query travels from input to final synthesized response.*

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as Async Pipeline
    participant Embed as bge-small (Local CPU)
    participant VecDB as sqlite-vec (Local)
    participant FlashRank as FlashRank (Local CPU)
    participant Ollama as Ollama Cloud (LLM)

    User->>Pipeline: "What are the core features of project X?"
    Pipeline->>Ollama: [Query Rewrite Task]
    Ollama-->>Pipeline: Returns Optimized Queries
    
    Pipeline->>Embed: Send Optimized Queries
    Embed-->>Pipeline: Returns Dense Vectors
    
    Pipeline->>VecDB: Retrieve top-K=100 chunks
    VecDB-->>Pipeline: Returns 100 Raw Chunks
    
    Pipeline->>FlashRank: Rerank Top 100
    FlashRank-->>Pipeline: Returns Top-K=15 Context Chunks
    
    Pipeline->>Ollama: Generate response using Top-K=15 Context
    Ollama-->>Pipeline: Streams Final Synthesis
    Pipeline->>User: Streams Final Answer
```

---

### 3. Memory Hierarchy Diagram
*Visualizing the cognitive architecture for temporal and static memory.*

```mermaid
graph BT
    L1["L1: Model Memory<br>(Parametric weights in Cloud LLM)"] 
    L2["L2: Working Memory<br>(Pipeline Context Window & State)"]
    L3["L3: Semantic System Memory<br>(sqlite-vec chunks & documents)"]
    L4["L4: User Preferences<br>(Versioned store - SQLite KV, user-editable)"]

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
    Query["User Query"] --> Embed["bge-small-en-v1.5<br>(Local CPU Embedding)"]
    
    Embed --> VecDB["sqlite-vec<br>(Dense Vector Search)"]
    
    VecDB --> FlashRank["FlashRank<br>(Local CPU Reranking)"]
    
    FlashRank --> Final["Final Context Payload"]
```

---

### 5. Linear Pipeline Orchestration
*The brutalist linear pipeline replacing the cyclical multi-agent framework.*

```mermaid
graph TD
    Start["User Query"] --> Embed["Local Embedding<br>(bge-small)"]
    Embed --> Retrieve["Dense Retrieval Phase<br>(sqlite-vec)"]
    
    Retrieve --> Rerank["FlashRank Reranking"]
    
    Rerank --> Generate["Final Generation (Ollama)"]
    
    Generate -->|API Failure / Timeout| Fallback["Graceful Degradation<br>(Surface Raw Evidence to User)"]
    
    classDef pipe fill:#44337a,stroke:#b794f4,color:#faf5ff
    class Embed,Retrieve,Rerank,Generate,Fallback pipe
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
    
    ChunkText --> BgeEmbed["bge-small-en-v1.5<br>(Local CPU Embedding)"]
    
    BgeEmbed -->|Dense Vectors| VecDB[("sqlite-vec")]
    ChunkText -->|Raw Text| SQL[("Document Store")]
    
    classDef pipe fill:#1a202c,stroke:#a0aec0,color:#f7fafc
    class Input,Cleaner,SemanticChunking,ChunkText,BgeEmbed pipe
```
