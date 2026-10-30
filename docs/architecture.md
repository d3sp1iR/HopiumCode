# Архитектура

```
 ИНДЕКСАЦИЯ (офлайн)
 data/repo ──> loader ──> chunker ──> embed ──> vectorstore (Chroma)
                              └────> lexical (BM25)
                 └──> graph.extract ──> graph.store (Kuzu | NetworkX)

 ЗАПРОС (онлайн)
 вопрос ──> orchestrator ──> retrieval.hybrid (вектор + BM25 + обход графа)
                │                      │
                │                      v
                │            prompts: обогащённый запрос ──> Ollama (LLM)
                │                                              │ tool_call
                │<──────── mcp_client <── MCP-сервер <─────────┘
                v
          verify_citations ──> ответ со ссылками [путь:строки]
```

Соответствие блокам схемы Гапанюка и владельцам: `OWNERS.md`.

## Схема потока данных (Mermaid)

```mermaid
flowchart LR
  subgraph IDX[Индексация, офлайн]
    R[data/repo] --> L[loader] --> C[chunker]
    C --> E[embed: Ollama] --> V[(Chroma)]
    C --> B[BM25]
    R --> X[graph.extract: ast] --> G[(Граф: NetworkX)]
  end
  subgraph Q[Запрос, онлайн]
    U[Вопрос] --> O[orchestrator]
    O --> H[retrieval.hybrid]
    H --> V
    H --> B
    H --> G
    H --> P[prompts: обогащённый запрос]
    P --> M[Ollama LLM]
    M -- tool_call --> MC[mcp_client] --> MS[MCP-сервер, только чтение] --> R
    MS --> G
    M --> VC[verify_citations]
    VC --> A[Ответ со ссылками]
  end
```
