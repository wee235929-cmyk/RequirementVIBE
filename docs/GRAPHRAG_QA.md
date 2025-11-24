# GraphRAG Question-and-Answer Logic

This document explains how the GraphRAG (Graph-based Retrieval Augmented Generation) system works in the ReqVIBE project for answering questions based on uploaded documents.

## Overview

GraphRAG is a document analysis and question-answering system that:
- Builds a knowledge graph from structured documents
- Uses semantic embeddings for intelligent document retrieval
- Enables context-aware question answering over document collections
- Falls back gracefully to normal chat if GraphRAG is unavailable or queries are not document-related

## Architecture

The GraphRAG system consists of several key components:

1. **Document Processing Pipeline** (`presentation/components/file_upload.py`)
2. **GraphRAG Index Building** (`infrastructure/graphrag/service.py`)
3. **Query Processing & Answering** (`infrastructure/graphrag/service.py` + `app.py`)
4. **Knowledge Graph Construction** (NetworkX-based)

## Workflow

### 1. Document Upload & Processing

**Location**: `presentation/components/file_upload.py`

Users upload documents (PDF, DOCX, ReqIF, or pre-processed JSON) through the sidebar:

```python
# Supported formats
- PDF files
- Word documents (.docx)
- ReqIF files (.reqif)
- Pre-processed JSON (vectorized documents)
```

**Processing Steps**:

1. **File Validation**: Validates file formats and enforces 10MB total size limit
2. **JSON Detection**: If JSON files are uploaded, they're validated as vectorized documents and used directly
3. **Unstructured API Processing**: Non-JSON files are processed using the Unstructured API to extract structured content
4. **Result Combination**: All results (JSON + processed) are combined into a unified document structure

**Key Functions**:
- `process_uploaded_files()`: Main entry point for processing uploaded files
- `is_valid_vectorized_json()`: Validates pre-processed JSON documents
- `combine_vectorized_results()`: Merges multiple document results

### 2. GraphRAG Index Construction

**Location**: `infrastructure/graphrag/service.py`

After documents are processed, a GraphRAG index is automatically built:

#### 2.1 Text Chunking

**Function**: `chunk_text(text, chunk_size=500, overlap=50)`

The system intelligently splits documents into overlapping chunks while preserving structure:

- **Priority 1**: Section headers (e.g., "6.2", "6.2.1 References")
- **Priority 2**: Paragraph boundaries (double newlines)
- **Priority 3**: Single newline breaks
- **Priority 4**: Sentence boundaries (periods)
- **Default**: Fixed-size chunks with overlap for context preservation

**Key Features**:
- Preserves document structure (section headers, paragraphs)
- Uses overlapping chunks (default: 100 chars overlap, 800 char chunks) for better context
- Maintains semantic boundaries to avoid cutting important information

#### 2.2 Entity & Relationship Extraction

**Function**: `extract_entities_and_relationships(text)`

Extracts structured information from text chunks:

**Entity Extraction**:
- Uses pattern matching to identify capitalized phrases (potential entities)
- Filters out common words (stop words)
- Identifies requirement patterns (e.g., "REQ-6.2", "Specification 3.1")

**Relationship Extraction**:
- Identifies relationships using common patterns:
  - "Entity1 is/has/contains Entity2"
  - "Entity1 shall/should/must [action] Entity2"
- Extracts relation verbs (is, are, has, have, contains, includes, requires, etc.)
- Returns tuples of (subject, relation, object)

**Note**: This is a simplified extraction. For production use, consider more sophisticated NER models.

#### 2.3 Knowledge Graph Construction

**Function**: `build_knowledge_graph(chunks, node_texts)`

Builds a directed graph using NetworkX:

**Graph Structure**:
- **Chunk Nodes**: Each text chunk becomes a node (type: "chunk")
- **Entity Nodes**: Extracted entities become nodes (type: "entity")
- **Edges**:
  - Chunk → Entity: "contains" relationship
  - Entity → Entity: Extracted relationships
  - Chunk → Chunk: "follows" relationship (for nearby chunks)

**Key Features**:
- Directed graph allows representing asymmetric relationships
- Nodes contain metadata (type, text, name)
- Edges contain relation types

#### 2.4 Embedding Generation

**Function**: `generate_embeddings(texts, model_name="all-MiniLM-L6-v2")`

Generates semantic embeddings for text chunks:

**Model**: Uses SentenceTransformers with `all-MiniLM-L6-v2` model (cached after first load)

**Features**:
- **Model Caching**: Embedding model is cached globally to avoid reloading
- **Fallback Support**: If SentenceTransformers is unavailable, uses hash-based pseudo-embeddings
- **Error Handling**: Gracefully falls back on errors

**Embedding Usage**:
- Enables semantic similarity search for document retrieval
- Maps chunk IDs to embedding vectors for similarity calculations

#### 2.5 Index Assembly

**Function**: `build_graphrag_index(structured_docs)`

Assembles all components into a `GraphRAGIndex` object:

**Index Components**:
- `chunks`: List of text chunks
- `graph`: NetworkX directed graph
- `node_embeddings`: Dictionary mapping chunk IDs to embedding vectors
- `node_texts`: Dictionary mapping node IDs to text content
- `metadata`: Statistics (total chunks, nodes, edges, documents)

**Storage**:
- Index is serialized to dictionary for session state storage
- Graph and embeddings are rebuilt on-demand (not serialized to save space)

### 3. Query Processing

**Location**: `app.py` (lines 328-369)

When a user submits a query, the system decides whether to use GraphRAG:

#### 3.1 Document-Related Query Detection

**Function**: `is_document_related_query(query)`

Determines if a query is likely related to uploaded documents:

**Detection Keywords**:
- Document-related: "document", "requirement", "specification", "spec", "req"
- Question patterns: "what does", "what is", "explain", "describe", "tell me about"
- Reference patterns: "according to", "in the document", "from the document"
- Query patterns: "what are", "list", "show me", "find", "where is"

**Decision Logic**:
1. If GraphRAG index exists AND query is document-related → Use GraphRAG
2. Otherwise → Use normal chat flow

#### 3.2 Index Reconstruction

Before querying, the index is reconstructed from session state:

```python
# Reconstruct from serialized data
index_data = st.session_state.graphrag_index
graphrag_index = GraphRAGIndex.from_dict(index_data)

# Rebuild graph and embeddings (not serialized)
if st.session_state.get("document_processing_results"):
    graphrag_index = build_graphrag_index(
        st.session_state.document_processing_results
    )
```

### 4. Document Retrieval

**Location**: `infrastructure/graphrag/service.py`

**Function**: `find_relevant_chunks(query, index, top_k=5)`

Uses a **hybrid retrieval approach** combining multiple signals:

#### 4.1 Semantic Similarity (Primary)

1. **Query Embedding**: Generates embedding for the user query
2. **Similarity Calculation**: Computes cosine similarity between query and all chunk embeddings
3. **Scoring**: Ranks chunks by semantic similarity

#### 4.2 Keyword Matching (Boost)

1. **Keyword Extraction**: Extracts keywords from query (removes stop words)
2. **Match Counting**: Counts keyword matches in each chunk
3. **Score Boost**: Adds up to 0.6 points for keyword matches

#### 4.3 Section Number Matching (Strong Boost)

1. **Pattern Detection**: Identifies section numbers in query (e.g., "6.2", "6.2.1")
2. **Strong Boost**: Adds 0.8 points if section number found in chunk

#### 4.4 Exact Phrase Matching (Medium Boost)

1. **Phrase Detection**: Checks if entire query phrase exists in chunk
2. **Score Boost**: Adds 0.5 points for exact phrase matches

#### 4.5 Top-K Selection

- Adjusts `top_k` based on document size:
  - Small documents (<20 chunks): 5 chunks
  - Medium documents (20-50 chunks): 8 chunks
  - Large documents (>50 chunks): 12 chunks
- Returns sorted list of (chunk_id, similarity_score) tuples

### 5. Context Extraction

**Function**: `extract_graph_context(chunk_ids, index, depth=1)`

Extracts additional context from the knowledge graph:

**Process**:
1. Traverses graph starting from relevant chunk nodes
2. Collects related entities (up to 5 entities)
3. Limits to depth=1 (direct neighbors only) to avoid excessive context
4. Returns minimal context string with entity names only

**Purpose**: Provides relationship information that might not be in the chunk text itself.

### 6. Answer Generation

**Location**: `infrastructure/graphrag/service.py`

**Function**: `answer_question_with_graphrag(query, index, llm_client, model)`

#### 6.1 Chunk Filtering

Filters retrieved chunks by relevance threshold:

- **Similarity Threshold**: 0.2 (semantic similarity)
- **Keyword Threshold**: 2.0 (keyword match score)
- **Fallback**: If no chunks pass threshold, uses keyword-only search

#### 6.2 Context Building

Builds prompt context with size limits:

**Constraints**:
- **MAX_CONTEXT_CHARS**: 4000 characters (~1000 tokens)
- **Prioritization**: Most relevant chunks added first
- **Truncation**: Last chunk truncated if needed to fit limit

**Context Structure**:
1. Most relevant chunks (sorted by score)
2. Graph context (related entities, limited to 500 chars)

#### 6.3 LLM Prompt Construction

**System Prompt**:
```
You are a helpful assistant that answers questions based on the provided document context.
Use the document context to answer the user's question accurately. If the context doesn't contain enough
information to answer the question, say so clearly.
```

**User Prompt Structure**:
```
Based on the following document context, please answer the user's question.

Document Context:
[Relevant chunks from documents]

[Graph context with related entities]

User Question: [user query]

Please provide a clear and accurate answer based on the document context.
```

#### 6.4 LLM Invocation & Error Handling

**Features**:
- **Timeout**: 120 seconds (2 minutes) for document Q&A
- **Retry Logic**: Up to 3 retries for connection errors with exponential backoff
- **Fallback Response**: If all retries fail, returns relevant chunks as fallback answer
- **Error Detection**: Identifies connection errors (network, timeout, aborted)

#### 6.5 Response Display

**Location**: `app.py` (lines 354-364)

The GraphRAG answer is displayed with Mermaid diagram support:

```python
with st.chat_message("assistant"):
    with st.spinner("Searching documents..."):
        graphrag_answer = answer_question_with_graphrag(...)
        # Display with Mermaid support
        from utils.renderers.mermaid import render_message_with_mermaid
        render_message_with_mermaid(graphrag_answer)
```

### 7. Fallback Behavior

**Graceful Degradation**:

1. **No Index**: If GraphRAG index not built → Normal chat flow
2. **Non-Document Query**: If query not document-related → Normal chat flow
3. **GraphRAG Failure**: If GraphRAG processing fails → Falls back to normal chat with error logging
4. **No Relevant Chunks**: Returns informative message asking user to rephrase

**Normal Chat Flow**: Uses conversation history and requirements management for standard Q&A.

## Key Data Structures

### GraphRAGIndex

```python
class GraphRAGIndex:
    graph: nx.DiGraph           # Knowledge graph
    node_embeddings: Dict       # Node ID -> embedding vector
    node_texts: Dict            # Node ID -> text content
    chunks: List[str]           # List of text chunks
    metadata: Dict              # Statistics and metadata
```

### Serialization

- **Serialized** (stored in session state): `chunks`, `node_texts`, `metadata`
- **Rebuilt on-demand**: `graph`, `node_embeddings` (to save storage space)

## Configuration & Parameters

### Chunking Parameters
- `chunk_size`: 800 characters (default)
- `overlap`: 100 characters (default)

### Retrieval Parameters
- `top_k`: 5-12 chunks (adaptive based on document size)
- `similarity_threshold`: 0.2
- `keyword_threshold`: 2.0

### Context Limits
- `MAX_CONTEXT_CHARS`: 4000 characters
- `max_entities`: 5 entities in graph context
- `graph_depth`: 1 (direct neighbors only)

### Embedding Model
- Model: `all-MiniLM-L6-v2` (SentenceTransformers)
- Dimension: 384 (standard for this model)
- Caching: Global model cache to avoid reloading

## Dependencies

### Required
- `networkx`: Graph construction and traversal
- `numpy`: Vector operations for embeddings (optional, with fallback)
- `sentence-transformers`: Semantic embeddings (optional, with fallback)

### Optional
- If `sentence-transformers` unavailable: Uses hash-based pseudo-embeddings
- If `numpy` unavailable: Uses simplified similarity calculations

## Performance Considerations

1. **Model Caching**: Embedding model cached after first load
2. **Lazy Reconstruction**: Graph and embeddings rebuilt only when needed
3. **Context Limits**: Strict limits prevent token overflow
4. **Chunk Prioritization**: Most relevant chunks included first
5. **Adaptive Retrieval**: `top_k` adjusts based on document size

## Error Handling

1. **Connection Errors**: Retry with exponential backoff (3 attempts)
2. **Missing Dependencies**: Graceful fallback to simplified methods
3. **Invalid Queries**: Informative error messages
4. **Empty Results**: Suggests rephrasing or checking document content

## Future Improvements

Potential enhancements for production use:

1. **Better Entity Extraction**: Use trained NER models instead of pattern matching
2. **Graph Traversal**: More sophisticated graph traversal algorithms
3. **Embedding Cache**: Cache document embeddings to avoid regeneration
4. **Hybrid Search**: Add BM25 or other keyword-based search methods
5. **Reranking**: Add a reranking step after initial retrieval
6. **Citation Tracking**: Track which chunks were used in answers for citations

## Related Files

- **Main Service**: `infrastructure/graphrag/service.py`
- **Integration**: `app.py` (lines 328-369)
- **File Upload**: `presentation/components/file_upload.py`
- **Document Processing**: `domain/documents/unstructured.py`
- **Response Rendering**: `utils/renderers/mermaid.py`

