"""
GraphRAG Service for Document Analysis

This service implements a GraphRAG (Graph-based Retrieval Augmented Generation) pipeline
that builds a knowledge graph from structured documents and enables question answering
over the graph using LLM integration.

Features:
- Text chunking for optimal processing
- Entity and relationship extraction
- Knowledge graph construction (using NetworkX)
- Node embeddings generation
- Question answering with graph context
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import networkx as nx

# Try to import optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from monitoring.langsmith import traceable

# Global cache for SentenceTransformer model to avoid reloading
_cached_embedding_model = None
_cached_model_name = None


class GraphRAGIndex:
    """Container for GraphRAG index data."""
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.node_embeddings = {}  # Node ID -> embedding vector
        self.node_texts = {}  # Node ID -> text content
        self.chunks = []  # List of text chunks
        self.metadata = {}  # Additional metadata
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize index to dictionary (for session state storage)."""
        return {
            'chunks': self.chunks,
            'node_texts': self.node_texts,
            'metadata': self.metadata,
            # Note: graph and embeddings are not serialized to save space
            # They will be rebuilt when needed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphRAGIndex':
        """Deserialize index from dictionary."""
        index = cls()
        index.chunks = data.get('chunks', [])
        index.node_texts = data.get('node_texts', {})
        index.metadata = data.get('metadata', {})
        return index


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks, preserving section headers and structure.
    
    This function tries to break at natural boundaries:
    1. Section headers (e.g., "6.2 References")
    2. Paragraph boundaries (double newlines)
    3. Sentence boundaries
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to find the best break point
        if end < len(text):
            # Priority 1: Section headers (e.g., "6.2", "6.2.1", "References")
            # Look for patterns like "6.2", "6.2.1", or standalone section titles
            section_pattern = r'\n\s*\d+\.\d+[\.\d]*\s+[A-Z]'  # Matches "6.2 Title"
            section_matches = list(re.finditer(section_pattern, chunk))
            if section_matches:
                # Use the last section header before the end
                last_section = section_matches[-1]
                if last_section.start() > chunk_size * 0.3:  # Only if past 30% of chunk
                    break_point = last_section.start()
                    chunk = chunk[:break_point]
                    end = start + break_point
                    chunks.append(chunk.strip())
                    start = end - overlap
                    continue
            
            # Priority 2: Paragraph boundaries (double newlines)
            last_paragraph = chunk.rfind('\n\n')
            if last_paragraph > chunk_size * 0.4:  # Only if past 40% of chunk
                chunk = chunk[:last_paragraph + 2]
                end = start + last_paragraph + 2
                chunks.append(chunk.strip())
                start = end - overlap
                continue
            
            # Priority 3: Single newline (line breaks)
            last_newline = chunk.rfind('\n')
            if last_newline > chunk_size * 0.5:  # Only if past 50% of chunk
                chunk = chunk[:last_newline + 1]
                end = start + last_newline + 1
                chunks.append(chunk.strip())
                start = end - overlap
                continue
            
            # Priority 4: Sentence boundaries
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.6:  # Only if past 60% of chunk
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
                chunks.append(chunk.strip())
                start = end - overlap
                continue
        
        # If no good break point found, use the chunk as-is
        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context
    
    return chunks


def extract_entities_and_relationships(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Extract entities and relationships from text using pattern matching.
    
    This is a simplified extraction. For production, consider using NER models
    or more sophisticated extraction methods.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (entities, relationships) where:
        - entities: List of entity strings
        - relationships: List of (subject, relation, object) tuples
    """
    entities = []
    relationships = []
    
    # Extract capitalized phrases (potential entities)
    # Pattern: Capitalized words (proper nouns, acronyms)
    entity_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
    potential_entities = re.findall(entity_pattern, text)
    
    # Filter out common words and keep meaningful entities
    common_words = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 
                   'What', 'Which', 'Who', 'How', 'Why', 'Should', 'Must', 
                   'Can', 'Will', 'May', 'Might', 'Could', 'Would'}
    entities = [e for e in potential_entities if e not in common_words and len(e) > 2]
    
    # Extract relationships using common patterns
    # Pattern: "Entity1 verb Entity2" or "Entity1 is/has/contains Entity2"
    relation_patterns = [
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:is|are|has|have|contains|includes|requires|needs|uses|implements)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:shall|should|must|will|can)\s+([a-z]+(?:\s+[a-z]+)*)',
    ]
    
    for pattern in relation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            subject = match.group(1)
            obj = match.group(2)
            relation = "related_to"  # Default relation type
            
            # Try to extract the verb as relation
            verb_match = re.search(r'\b(is|are|has|have|contains|includes|requires|needs|uses|implements|shall|should|must|will|can)\b', 
                                 match.group(0), re.IGNORECASE)
            if verb_match:
                relation = verb_match.group(1).lower()
            
            if subject and obj and subject != obj:
                relationships.append((subject, relation, obj))
    
    # Also extract requirement-like patterns
    req_pattern = r'(?:requirement|req|specification|spec)\s+([A-Z0-9]+(?:\.[A-Z0-9]+)*)'
    req_matches = re.findall(req_pattern, text, re.IGNORECASE)
    entities.extend([f"REQ-{req}" for req in req_matches])
    
    return list(set(entities)), relationships


def build_knowledge_graph(chunks: List[str], node_texts: Dict[str, str]) -> nx.DiGraph:
    """
    Build a knowledge graph from text chunks.
    
    Args:
        chunks: List of text chunks
        node_texts: Dictionary mapping node IDs to text content
        
    Returns:
        NetworkX directed graph
    """
    graph = nx.DiGraph()
    
    # Add chunk nodes
    for i, chunk in enumerate(chunks):
        node_id = f"chunk_{i}"
        graph.add_node(node_id, type="chunk", text=chunk)
    
    # Extract entities and relationships from each chunk
    all_entities = set()
    all_relationships = []
    
    for i, chunk in enumerate(chunks):
        entities, relationships = extract_entities_and_relationships(chunk)
        all_entities.update(entities)
        all_relationships.extend(relationships)
        
        # Add entities as nodes
        for entity in entities:
            entity_id = f"entity_{hash(entity) % 10000}"
            if not graph.has_node(entity_id):
                graph.add_node(entity_id, type="entity", name=entity)
            # Link chunk to entity
            graph.add_edge(f"chunk_{i}", entity_id, relation="contains")
        
        # Add relationships as edges
        for subj, rel, obj in relationships:
            subj_id = f"entity_{hash(subj) % 10000}"
            obj_id = f"entity_{hash(obj) % 10000}"
            
            # Ensure nodes exist
            if not graph.has_node(subj_id):
                graph.add_node(subj_id, type="entity", name=subj)
            if not graph.has_node(obj_id):
                graph.add_node(obj_id, type="entity", name=obj)
            
            # Add relationship edge
            graph.add_edge(subj_id, obj_id, relation=rel)
    
    # Add semantic similarity edges between chunks (simplified)
    # In production, use embeddings to find similar chunks
    for i in range(len(chunks)):
        for j in range(i + 1, min(i + 3, len(chunks))):  # Connect nearby chunks
            if not graph.has_edge(f"chunk_{i}", f"chunk_{j}"):
                graph.add_edge(f"chunk_{i}", f"chunk_{j}", relation="follows")
    
    return graph


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, List[float]]:
    """
    Generate embeddings for texts using sentence transformers.
    
    The model is cached after the first load to avoid reloading on every call,
    which significantly improves performance and avoids repeated network requests.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the sentence transformer model
        
    Returns:
        Dictionary mapping text index to embedding vector
    """
    global _cached_embedding_model, _cached_model_name
    
    embeddings = {}
    
    if not HAS_SENTENCE_TRANSFORMERS:
        # Fallback: simple hash-based "embeddings" (not real embeddings)
        for i, text in enumerate(texts):
            # Use a simple hash-based representation
            hash_val = hash(text)
            # Create a pseudo-embedding vector
            pseudo_embedding = [float((hash_val >> j) & 0xFF) / 255.0 for j in range(0, 128, 8)]
            embeddings[str(i)] = pseudo_embedding
        return embeddings
    
    try:
        # Load model only if not cached or if model name changed
        if _cached_embedding_model is None or _cached_model_name != model_name:
            print(f"Loading embedding model: {model_name} (this may take a moment on first use)...")
            _cached_embedding_model = SentenceTransformer(model_name)
            _cached_model_name = model_name
            print(f"Model {model_name} loaded and cached successfully.")
        
        # Use cached model
        embedding_vectors = _cached_embedding_model.encode(texts, show_progress_bar=False)
        
        for i, embedding in enumerate(embedding_vectors):
            embeddings[str(i)] = embedding.tolist()
    except Exception as e:
        # Fallback on error
        print(f"Warning: Embedding generation failed: {e}. Using fallback.")
        # Clear cache on error so it can retry next time
        _cached_embedding_model = None
        _cached_model_name = None
        for i, text in enumerate(texts):
            hash_val = hash(text)
            pseudo_embedding = [float((hash_val >> j) & 0xFF) / 255.0 for j in range(0, 128, 8)]
            embeddings[str(i)] = pseudo_embedding
    
    return embeddings


def process_uploaded_documents(structured_docs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process structured documents from Unstructured API into chunks.
    
    Args:
        structured_docs: Output from process_multiple_documents()
        
    Returns:
        Dictionary with processed document data including chunks
    """
    all_chunks = []
    node_texts = {}
    
    for doc in structured_docs.get('documents', []):
        doc_name = doc.get('filename', 'unknown')
        elements = doc.get('elements', [])
        
        # Extract text from elements
        doc_text_parts = []
        for element in elements:
            if isinstance(element, dict):
                text = element.get('text', '')
                element_type = element.get('type', 'unknown')
                if text:
                    doc_text_parts.append(f"[{element_type}] {text}")
        
        # Combine all text
        full_text = "\n\n".join(doc_text_parts)
        
        # Chunk the text with larger size to preserve context better
        # Use larger chunks for better context preservation, especially for section headers
        chunks = chunk_text(full_text, chunk_size=800, overlap=100)
        
        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_name}_chunk_{i}"
            all_chunks.append(chunk)
            node_texts[chunk_id] = chunk
    
    return {
        'chunks': all_chunks,
        'node_texts': node_texts,
        'total_chunks': len(all_chunks),
        'documents': structured_docs.get('documents', [])
    }


def build_graphrag_index(structured_docs: Dict[str, Any]) -> GraphRAGIndex:
    """
    Build a complete GraphRAG index from structured documents.
    
    Args:
        structured_docs: Output from process_multiple_documents()
        
    Returns:
        GraphRAGIndex object containing graph, embeddings, and metadata
    """
    index = GraphRAGIndex()
    
    # Process documents into chunks
    processed = process_uploaded_documents(structured_docs)
    index.chunks = processed['chunks']
    index.node_texts = processed['node_texts']
    
    # Build knowledge graph
    index.graph = build_knowledge_graph(index.chunks, index.node_texts)
    
    # Generate embeddings for chunks
    if index.chunks:
        embeddings = generate_embeddings(index.chunks)
        # Map embeddings to chunk indices
        for i, chunk in enumerate(index.chunks):
            chunk_id = f"chunk_{i}"
            if str(i) in embeddings:
                index.node_embeddings[chunk_id] = embeddings[str(i)]
    
    # Store metadata
    index.metadata = {
        'total_chunks': len(index.chunks),
        'total_nodes': index.graph.number_of_nodes(),
        'total_edges': index.graph.number_of_edges(),
        'documents': processed.get('documents', [])
    }
    
    return index


def find_relevant_chunks(query: str, index: GraphRAGIndex, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Find the most relevant chunks for a query using embeddings and keyword matching.
    
    Uses a hybrid approach:
    1. Semantic similarity via embeddings
    2. Keyword/term matching as a boost
    3. Section header matching (e.g., "6.2 References")
    
    Args:
        query: User query string
        index: GraphRAGIndex object
        top_k: Number of top chunks to return (increased for larger documents)
        
    Returns:
        List of (chunk_id, similarity_score) tuples
    """
    if not index.chunks:
        return []
    
    # Adjust top_k based on document size, but keep it reasonable
    total_chunks = len(index.chunks)
    if total_chunks > 50:
        top_k = int(min(top_k * 1.5, 12))  # Slightly more for larger documents, but capped
    elif total_chunks > 20:
        top_k = int(min(top_k * 1.2, 8))  # Moderate increase
    else:
        top_k = int(top_k)  # Ensure it's an integer
    
    # Extract keywords from query (simple approach)
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                  'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about'}
    query_keywords = query_words - stop_words
    
    # Extract section numbers/headers from query (e.g., "6.2", "References")
    section_pattern = r'\b\d+\.\d+[\.\d]*\b'  # Matches "6.2", "6.2.1", etc.
    section_numbers = re.findall(section_pattern, query)
    
    similarities = []
    
    # If we have embeddings, use semantic similarity
    if index.node_embeddings:
        # Generate query embedding
        query_embeddings = generate_embeddings([query])
        if query_embeddings and '0' in query_embeddings:
            query_embedding = query_embeddings['0']
            
            # Calculate similarities (cosine similarity)
            for i, chunk in enumerate(index.chunks):
                chunk_id = f"chunk_{i}"
                if chunk_id in index.node_embeddings:
                    chunk_embedding = index.node_embeddings[chunk_id]
                    
                    # Calculate cosine similarity
                    if HAS_NUMPY:
                        try:
                            dot_product = np.dot(query_embedding, chunk_embedding)
                            norm_query = np.linalg.norm(query_embedding)
                            norm_chunk = np.linalg.norm(chunk_embedding)
                            similarity = dot_product / (norm_query * norm_chunk + 1e-8)
                        except:
                            similarity = 0.0
                    else:
                        # Fallback: simple dot product (not normalized)
                        similarity = sum(a * b for a, b in zip(query_embedding, chunk_embedding)) / 100.0
                    
                    similarities.append((chunk_id, similarity))
    
    # If no embeddings or empty results, initialize with zero scores
    if not similarities:
        for i in range(len(index.chunks)):
            chunk_id = f"chunk_{i}"
            similarities.append((chunk_id, 0.0))
    
    # Boost scores with keyword matching
    for i, (chunk_id, base_score) in enumerate(similarities):
        chunk_text = index.chunks[i].lower()
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text))
        
        # Keyword matching boost
        keyword_matches = len(query_keywords & chunk_words)
        keyword_boost = min(keyword_matches * 0.2, 0.6)  # Max 0.6 boost
        
        # Section number matching boost (stronger)
        section_boost = 0.0
        for section_num in section_numbers:
            if section_num in chunk_text:
                section_boost = 0.8  # Strong boost for section number matches
                break
        
        # Exact phrase matching boost
        phrase_boost = 0.0
        if query_lower in chunk_text:
            phrase_boost = 0.5
        
        # Combine scores
        final_score = base_score + keyword_boost + section_boost + phrase_boost
        similarities[i] = (chunk_id, final_score)
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def extract_graph_context(chunk_ids: List[str], index: GraphRAGIndex, depth: int = 1) -> str:
    """
    Extract context from the knowledge graph around relevant chunks.
    
    This function extracts only entity relationships, NOT chunk content
    (chunks are already included in the main context).
    
    Args:
        chunk_ids: List of relevant chunk node IDs
        index: GraphRAGIndex object
        depth: How many hops to traverse in the graph (reduced to 1 to limit context)
        
    Returns:
        Context string with relevant graph information (entities only)
    """
    context_parts = []
    visited_entities = set()
    max_entities = 5  # Limit number of entities to avoid excessive context
    
    for chunk_id in chunk_ids:
        if chunk_id not in index.graph or len(visited_entities) >= max_entities:
            continue
        
        # Traverse graph to find related entities (NOT chunks - those are already included)
        try:
            # Only get direct neighbors (depth=1) to limit context
            neighbors = list(index.graph.neighbors(chunk_id))
            
            for neighbor in neighbors:
                if len(visited_entities) >= max_entities:
                    break
                    
                if neighbor in visited_entities:
                    continue
                
                # Get node attributes
                if neighbor in index.graph.nodes:
                    node_data = index.graph.nodes[neighbor]
                    node_type = node_data.get('type', 'unknown')
                    if node_type == 'entity':
                        entity_name = node_data.get('name', '')
                        if entity_name and entity_name not in visited_entities:
                            visited_entities.add(entity_name)
                            context_parts.append(f"Related Entity: {entity_name}")
        except:
            pass  # Ignore graph traversal errors
    
    # Return minimal context (just entity names, not full chunks)
    if context_parts:
        return "\n".join(context_parts[:max_entities])
    return ""


@traceable(name="graphrag_answer", run_type="llm")
def answer_question_with_graphrag(
    query: str, 
    index: GraphRAGIndex,
    llm_client,
    model: str = "deepseek-chat"
) -> str:
    """
    Answer a question using GraphRAG index.
    
    Args:
        query: User question
        index: GraphRAGIndex object
        llm_client: LLM client instance
        model: Model name to use
        
    Returns:
        Answer string
    """
    if not index.chunks:
        return "No document context available. Please upload and process documents first."
    
    # Find relevant chunks with hybrid search (start with reasonable number)
    relevant_chunks = find_relevant_chunks(query, index, top_k=8)  # Reduced from 10 to 8
    
    # Filter chunks by similarity threshold - only keep chunks with meaningful relevance
    # Threshold: 0.2 for semantic similarity (stricter), or 2.0+ for keyword matches
    similarity_threshold = 0.2  # Increased from 0.15 to be more selective
    keyword_threshold = 2.0
    
    filtered_chunks = []
    for chunk_id, score in relevant_chunks:
        # Accept if semantic similarity is above threshold OR keyword match is strong
        if score >= similarity_threshold or (score >= keyword_threshold and score < 10):  # keyword scores are typically 1-10
            filtered_chunks.append((chunk_id, score))
    
    # If no chunks pass the threshold, try keyword-based search as fallback
    if not filtered_chunks:
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                      'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about'}
        query_keywords = query_keywords - stop_words
        
        # Only proceed if we have meaningful keywords
        if not query_keywords:
            return "Could not find relevant information in the documents. Please try rephrasing your question or check if the content exists in the uploaded documents."
        
        keyword_matches = []
        for i, chunk in enumerate(index.chunks):
            chunk_lower = chunk.lower()
            # Count keyword matches
            matches = sum(1 for keyword in query_keywords if keyword in chunk_lower)
            if matches >= 2:  # Require at least 2 keyword matches
                # Also check for exact phrase matches
                if query_lower in chunk_lower:
                    matches += 5  # Strong boost for exact phrase
                keyword_matches.append((f"chunk_{i}", float(matches)))
        
        if keyword_matches:
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            # Limit to top 5 keyword matches and filter by threshold
            filtered_chunks = [(cid, score) for cid, score in keyword_matches[:5] if score >= keyword_threshold]
    
    if not filtered_chunks:
        return "Could not find relevant information in the documents. Please try rephrasing your question or check if the content exists in the uploaded documents."
    
    # Use filtered chunks
    relevant_chunks = filtered_chunks
    
    # Extract graph context (limited to avoid excessive content)
    chunk_ids = [chunk_id for chunk_id, _ in relevant_chunks]
    graph_context = extract_graph_context(chunk_ids, index, depth=1)  # Reduced depth from 2 to 1
    
    # Build prompt with context, with size limits
    MAX_CONTEXT_CHARS = 4000  # Limit total context to ~1000 tokens (rough estimate: 1 token ≈ 4 chars)
    context_parts = []
    total_chars = 0
    
    # Sort chunks by relevance score (highest first) to prioritize most relevant content
    sorted_chunks = sorted(relevant_chunks, key=lambda x: x[1], reverse=True)
    
    for chunk_id, score in sorted_chunks:
        if '_' in chunk_id and chunk_id.split('_')[1].isdigit():
            chunk_idx = int(chunk_id.split('_')[1])
            if chunk_idx < len(index.chunks):
                chunk_text = index.chunks[chunk_idx]
                chunk_size = len(chunk_text)
                
                # Check if adding this chunk would exceed the limit
                if total_chars + chunk_size > MAX_CONTEXT_CHARS:
                    # Truncate the last chunk if needed
                    remaining = int(MAX_CONTEXT_CHARS - total_chars)
                    if remaining > 200:  # Only add if we have meaningful space left
                        chunk_text = chunk_text[:remaining] + "..."
                        context_parts.append(chunk_text)
                    break
                
                context_parts.append(chunk_text)
                total_chars += chunk_size
    
    context_text = "\n\n".join(context_parts)
    
    # Limit graph context size as well
    if len(graph_context) > 500:
        graph_context = graph_context[:500] + "..."
    
    system_prompt = """You are a helpful assistant that answers questions based on the provided document context.
Use the document context to answer the user's question accurately. If the context doesn't contain enough
information to answer the question, say so clearly."""
    
    user_prompt = f"""Based on the following document context, please answer the user's question.

Document Context:
{context_text}

{graph_context}

User Question: {query}

Please provide a clear and accurate answer based on the document context."""
    
    # Retry logic for connection errors
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Call LLM with timeout
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=120  # 2 minutes timeout for document Q&A
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            # Check if it's a connection error that might be retryable
            is_connection_error = any(keyword in error_str for keyword in [
                'connection', 'aborted', 'reset', 'timeout', 'network', 
                '远程主机', '连接', '超时'
            ])
            
            # If it's the last attempt or not a connection error, provide fallback
            if attempt == max_retries - 1 or not is_connection_error:
                # Provide a helpful fallback response with relevant document chunks
                fallback_answer = (
                    "I encountered a connection issue while generating an answer. "
                    "However, I found the following relevant information from your documents:\n\n"
                )
                
                # Add relevant chunks as fallback
                for i, (chunk_id, similarity) in enumerate(relevant_chunks[:3], 1):
                    chunk_idx = int(chunk_id.split('_')[1]) if '_' in chunk_id and chunk_id.split('_')[1].isdigit() else 0
                    if chunk_idx < len(index.chunks):
                        chunk_text = index.chunks[chunk_idx]
                        # Truncate long chunks for readability
                        if len(chunk_text) > 500:
                            chunk_text = chunk_text[:500] + "..."
                        fallback_answer += f"**Relevant Content {i}:**\n{chunk_text}\n\n"
                
                fallback_answer += (
                    "\n**Note:** There was a connection error when trying to generate a comprehensive answer. "
                    "Please try again, or check your network connection and API configuration."
                )
                
                return fallback_answer
            
            # Wait before retrying (exponential backoff)
            wait_time = retry_delay * (2 ** attempt)
            print(f"Connection error on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # This should never be reached, but just in case
    return "Error: Unable to generate answer after multiple retry attempts. Please check your network connection and try again."


def is_document_related_query(query: str) -> bool:
    """
    Determine if a query is likely related to uploaded documents.
    
    Args:
        query: User query string
        
    Returns:
        True if query seems document-related
    """
    query_lower = query.lower()
    
    # Keywords that suggest document-related queries
    doc_keywords = [
        'document', 'requirement', 'specification', 'spec', 'req',
        'what does', 'what is', 'explain', 'describe', 'tell me about',
        'according to', 'in the document', 'from the document',
        'what are', 'list', 'show me', 'find', 'where is'
    ]
    
    return any(keyword in query_lower for keyword in doc_keywords)

