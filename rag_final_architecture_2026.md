# Phase 4 — Final Architecture Maps
## Visual Blueprint for the 2026 Thin-Client RAG System

> [!NOTE]
> Below are the 7 requested architectural diagrams representing the definitive state of your 2026 personalized RAG system (16GB RAM / 2GB VRAM constraint). These diagrams map the "Thin-Client" paradigm, where orchestration runs locally but heavy matrix compute is offloaded to the cloud.

---

### 1. Full System Architecture Diagram
*A high-level view of the entire local-to-cloud split architecture.*

```mermaid
graph TD
    subgraph "Local Machine (16GB RAM / 2GB VRAM)"
        UI["User Interface (CLI/Web)"]
        Orchestrator["LangGraph Orchestrator (Python)"]
        LocalEmbed["Jina v5-text-small (2GB VRAM)"]
        
        subgraph "Local Storage Layer"
            VecDB["sqlite-vec (Vector DB)"]
            DocStore["SQLite (Raw Docs)"]
            StateDB["PostgresSaver (Agent State)"]
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
    Orchestrator <--> Cohere
    
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
    participant Jina as Jina (Local Embed)
    participant VecDB as sqlite-vec (Local)
    participant Cohere as Cohere Rerank (Cloud)
    participant Ollama as Ollama Cloud (LLM)

    User->>LangGraph: "What are the core features of project X?"
    LangGraph->>Ollama: [Query Routing/Rewrite Task]
    Ollama-->>LangGraph: Returns Optimized Queries
    
    LangGraph->>Jina: Send Optimized Queries
    Jina-->>LangGraph: Returns Dense Vectors
    
    LangGraph->>VecDB: Retrieve top-K=100 chunks
    VecDB-->>LangGraph: Returns 100 Raw Chunks
    
    LangGraph->>Cohere: Send 100 Chunks + Query for Reranking
    Cohere-->>LangGraph: Returns Top-K=5 Reranked Chunks
    
    LangGraph->>Ollama: Generate response using Top-K=5 Context
    Ollama-->>LangGraph: Streams Final Synthesis
    LangGraph->>User: Streams Final Answer
```

---

### 3. Memory Hierarchy Diagram
*Visualizing the CoALA cognitive architecture for temporal and static memory.*

```mermaid
graph BT
    L1["L1: Model Memory<br>(Parametric weights in Cloud LLM)"] 
    L2["L2: Working Memory<br>(LangGraph Context Window & State)"]
    L3["L3: Semantic System Memory<br>(sqlite-vec chunks & documents)"]
    L4["L4: Procedural / Episodic Memory<br>(Graphiti temporal graphs for agent rules)"]

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
    Query["User Query"] --> Router{"Complexity Router<br>(Small LLM call)"}
    
    Router -->|Simple Fact| DirectSearch["Standard Search"]
    Router -->|Complex / Multi-hop| Decomp["Query Decomposition"]
    
    Decomp --> SubQ1["Sub-Query 1"]
    Decomp --> SubQ2["Sub-Query 2"]
    
    DirectSearch --> Hybrid["Hybrid Retrieval<br>(BM25 + Dense)"]
    SubQ1 --> Hybrid
    SubQ2 --> Hybrid
    
    Hybrid --> RRF["Reciprocal Rank Fusion (RRF)<br>Merge Results"]
    RRF --> Rerank["Cross-Encoder Reranker<br>(Cohere)"]
    Rerank --> Threshold{"Relevance > 0.8?"}
    
    Threshold -->|Yes| Final["Final Context Payload"]
    Threshold -->|No| WebSearch["Trigger Web Search Tool"]
```

---

### 5. Agent Communication Flow
*The LangGraph cyclic state machine.*

```mermaid
graph TD
    Supervisor{"Supervisor Agent"}
    
    ResearchAgent["Research Sub-Agent<br>(Vector Search)"]
    WebAgent["Web Search Sub-Agent<br>(Tavily/Firecrawl)"]
    CodeAgent["Code Sub-Agent<br>(AST Tree-sitter Search)"]
    Critic["Evaluator / Critic"]
    
    Supervisor -->|Routes to| ResearchAgent
    Supervisor -->|Routes to| WebAgent
    Supervisor -->|Routes to| CodeAgent
    
    ResearchAgent -->|Returns data| Supervisor
    WebAgent -->|Returns data| Supervisor
    CodeAgent -->|Returns data| Supervisor
    
    Supervisor -->|Requests Review| Critic
    Critic -->|Passes| End["Final Generation"]
    Critic -->|Fails: 'Needs more info'| Supervisor
    
    classDef agent fill:#44337a,stroke:#b794f4,color:#faf5ff
    class Supervisor,ResearchAgent,WebAgent,CodeAgent,Critic agent
```

---

### 6. Storage Architecture Map
*How data is physically stored locally vs transiently.*

```mermaid
graph LR
    subgraph "Local Disk (Persistent)"
        VDB[("sqlite-vec<br>(Embeddings & Metadata)")]
        SQL[("SQLite<br>(Raw Doc Markdown)")]
        PG[("PostgresSaver<br>(LangGraph Checkpoints)")]
    end
    
    subgraph "System RAM (16GB) - Transient"
        InMemCache["Redis / Semantic Cache<br>(Recent Queries)"]
        ActiveGraph["Active LangGraph State"]
    end
    
    subgraph "GPU VRAM (2GB) - Transient"
        EmbModel["Jina v5-text-small Weights"]
    end

    ActiveGraph -->|Reads/Writes| PG
    ActiveGraph -->|Reads| SQL
    ActiveGraph -->|Searches| VDB
    ActiveGraph -->|Checks| InMemCache
```

---

### 7. Ingestion Pipeline Map
*How complex files are processed before hitting the database.*

```mermaid
graph TD
    Input["Raw Files<br>(PDF, DOCX, Code)"]
    
    Input --> LlamaParse{"LlamaParse API<br>(Cloud VLM Extraction)"}
    
    LlamaParse -->|Raw Markdown + Tables| Cleaner["Local Text Cleaner<br>(Regex / Normalization)"]
    
    Cleaner --> LateChunking["Late Chunking Logic<br>(Jina)"]
    
    LateChunking --> ChunkText["Context-Aware Chunks"]
    
    ChunkText --> JinaEmbed["Jina Local Embedding Model<br>(2GB VRAM)"]
    
    JinaEmbed -->|Dense Vectors| VecDB[("sqlite-vec")]
    ChunkText -->|Raw Text| SQL[("Document Store")]
    
    classDef pipe fill:#1a202c,stroke:#a0aec0,color:#f7fafc
    class Input,Cleaner,LateChunking,ChunkText,JinaEmbed pipe
```
