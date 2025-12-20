import os
import json
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = "./vector_db"
METADATA_FILE = "metadata.json"
TOP_K_PER_QUERY = 5  # Get 5 results per query (10 total)
CONVERSATION_MEMORY_SIZE = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL = "gemini-2.5-flash"


def load_gemini_client():
    """Initialize and return Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in .env file")
    
    return genai.Client(api_key=api_key)


def load_embedding_model():
    """Load and return the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def load_vector_database():
    """Load FAISS index and metadata."""
    index_path = os.path.join(DB_PATH, "vector_index.faiss")
    metadata_path = os.path.join(DB_PATH, METADATA_FILE)
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise Exception("Database not found. Run embed_data.py first.")
    
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return index, metadata


def get_surrounding_chunks(metadata, filename, chunk_index, context_range=3):
    """Get surrounding chunks (previous and next) from the same file."""
    surrounding_chunks = []
    
    for meta in metadata:
        if meta['filename'] == filename:
            current_idx = meta['chunk_index']
            if (chunk_index - context_range <= current_idx < chunk_index) or \
               (chunk_index < current_idx <= chunk_index + context_range):
                surrounding_chunks.append({
                    'text': meta['text'],
                    'filename': meta['filename'],
                    'chunk_index': meta['chunk_index']
                })
    
    return surrounding_chunks


def build_conversation_history(messages):
    """Build formatted conversation history from message list."""
    if not messages:
        return ""
    
    # Get last N exchanges for context
    recent_messages = messages[-CONVERSATION_MEMORY_SIZE * 2:] if len(messages) > CONVERSATION_MEMORY_SIZE * 2 else messages
    
    history_text = "Previous Conversation:\n"
    for msg in recent_messages:
        if msg["role"] == "user":
            history_text += f"\nUser: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"Assistant: {msg['content']}\n"
    
    return history_text


def reformulate_query(user_query, conversation_messages, gemini_client):
    """
    Reformulate the user query into a search-optimized text for embedding generation.
    This creates a concise text that captures full context for RAG retrieval.
    """
    # If no conversation history, return original query
    if not conversation_messages:
        return user_query
    
    # Build conversation history for reformulation
    conversation_history = build_conversation_history(conversation_messages)
    
    # Create prompt for query reformulation
    reformulation_prompt = f"""You are a search query optimization assistant. Given a conversation history and a new user query, create a concise search text that captures the complete context and intent.

{conversation_history}

Current User Query: {user_query}

Task: Create a search-optimized text (NOT a conversational query) that will be used to generate embeddings for RAG (Retrieval-Augmented Generation) matching. This text should:

1. Include all relevant context from the conversation history if required
2. Be concise and focused on key information needs (2-3 sentences max)
3. Include specific details mentioned in context (like course names, departments, semesters, etc.)
4. Be optimized for semantic similarity matching with document chunks
5. NOT mention "NIT Warangal" unless explicitly asked
6. If information type is mentioned (course details, credits, syllabus, course articulation matrix, textbooks, resources, etc.), explicitly include those keywords for better matching

Important: This is NOT a conversational question - it's a search text for embedding generation. Focus on keywords and context that will match relevant document sections.
This will be used to find relevant documents for answering the user's question so answer accordingly.

Examples:
- If user asks "what about mechanical?" after asking about CSE courses, output: "4th semester mechanical department courses"
- If user asks "tell me more" after discussing a syllabus, output: "detailed syllabus course structure prerequisites textbooks"

Search-optimized text (output only the text, nothing else):"""
    
    try:
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=reformulation_prompt
        )
        
        reformulated = response.text.strip()
        
        # Remove quotes if present
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]
        if reformulated.startswith("'") and reformulated.endswith("'"):
            reformulated = reformulated[1:-1]
        
        print(f"[Query Reformulation]")
        print(f"  Original: {user_query}")
        print(f"  Reformulated for RAG: {reformulated}\n")
        
        return reformulated
    
    except Exception as e:
        print(f"Query reformulation failed, using original query: {e}")
        return user_query


def retrieve_context_dual_query(original_query, reformulated_query, index, metadata, embedding_model):
    """
    Retrieve context using BOTH original and reformulated queries.
    Gets top 5 matches from each query (10 total) with surrounding chunks.
    """
    all_retrieved_chunks = []
    
    print("[Dual Query Retrieval]")
    
    # Query 1: Original query
    print(f"  Query 1 (Original): {original_query}")
    original_embedding = embedding_model.encode(
        [original_query],
        convert_to_numpy=True
    ).astype('float32')
    
    distances1, indices1 = index.search(original_embedding, TOP_K_PER_QUERY)
    print(f"    Found {len(indices1[0])} matches")
    
    # Query 2: Reformulated query
    print(f"  Query 2 (Reformulated): {reformulated_query}")
    reformulated_embedding = embedding_model.encode(
        [reformulated_query],
        convert_to_numpy=True
    ).astype('float32')
    
    distances2, indices2 = index.search(reformulated_embedding, TOP_K_PER_QUERY)
    print(f"    Found {len(indices2[0])} matches\n")
    
    # Combine indices from both queries
    combined_indices = list(indices1[0]) + list(indices2[0])
    
    # Process each match
    seen_chunks = set()
    
    for idx in combined_indices:
        main_meta = metadata[idx]
        filename = main_meta['filename']
        chunk_index = main_meta['chunk_index']
        main_text = main_meta['text']
        
        # Create unique identifier for this chunk
        chunk_id = f"{filename}_{chunk_index}"
        
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            
            # Add main chunk with metadata
            all_retrieved_chunks.append({
                'text': main_text,
                'filename': filename,
                'chunk_index': chunk_index,
                'total_chunks': main_meta.get('total_chunks', 'N/A'),
                'is_main': True
            })
            
            # Get surrounding chunks (3 before + 3 after)
            surrounding = get_surrounding_chunks(metadata, filename, chunk_index, context_range=3)
            
            for surr_chunk in surrounding:
                surr_id = f"{surr_chunk['filename']}_{surr_chunk['chunk_index']}"
                if surr_id not in seen_chunks:
                    seen_chunks.add(surr_id)
                    all_retrieved_chunks.append({
                        'text': surr_chunk['text'],
                        'filename': surr_chunk['filename'],
                        'chunk_index': surr_chunk['chunk_index'],
                        'total_chunks': 'N/A',
                        'is_main': False
                    })
    
    print(f"[Context Collection]")
    print(f"  Total unique chunks retrieved: {len(all_retrieved_chunks)}")
    print(f"  Main matches: {sum(1 for c in all_retrieved_chunks if c['is_main'])}")
    print(f"  Surrounding context: {sum(1 for c in all_retrieved_chunks if not c['is_main'])}\n")
    
    return all_retrieved_chunks


def format_context_with_metadata(chunks):
    """Format chunks with metadata information for better context."""
    formatted_context = []
    
    current_file = None
    for chunk in chunks:
        # Add file header when switching files
        if chunk['filename'] != current_file:
            current_file = chunk['filename']
            formatted_context.append(f"\n{'='*60}")
            formatted_context.append(f"SOURCE: {chunk['filename']}")
            formatted_context.append(f"{'='*60}\n")
        
        # Add chunk with position info
        chunk_info = f"[Chunk {chunk['chunk_index'] + 1}"
        if chunk.get('total_chunks') != 'N/A':
            chunk_info += f" of {chunk['total_chunks']}"
        chunk_info += "]"
        
        formatted_context.append(f"{chunk_info}\n{chunk['text']}\n")
    
    return "\n".join(formatted_context)


def generate_answer(original_query, reformulated_query, context_chunks, conversation_history, gemini_client):
    """Generate answer using Gemini with both queries, structured context and conversation history."""
    # Format context with metadata
    document_context = format_context_with_metadata(context_chunks)
    
    # Build prompt with both queries
    if conversation_history:
        prompt = f"""You are a helpful college chatbot assistant for NIT Warangal. Answer the user's question based on the provided context and conversation history.

{conversation_history}

User's Original Question: {original_query}
Search Query Used for Retrieval: {reformulated_query}

Relevant Information from Documents (with source files):
{document_context}

Instructions:
- Answer the ORIGINAL question naturally, considering conversation history
- Use the retrieved document context to provide accurate information
- You can see which file each piece of information comes from (marked as SOURCE)
- For normal queries, reply as a helpful assistant
- Do not mention anything that is out of context of the query
- If no context matches the question and no info in previous chats, respond with "I don't have information on that currently"
- You may mention the source file name if it adds value to the answer but not too frequently (e.g., "According to the CSE curriculum document...")

Answer:"""
    else:
        prompt = f"""You are a helpful college chatbot assistant for NIT Warangal. Answer the user's question based on the provided context.

User's Question: {original_query}

Relevant Information from Documents (with source files):
{document_context}

Instructions:
- Provide accurate information based on the document context
- You can see which file each piece of information comes from (marked as SOURCE)
- If no relevant context is found, respond with "I don't have information on that currently"
- You may mention the source file name if it adds value to the answer

Answer:"""
    
    # Generate response
    response = gemini_client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    
    return response.text


def query_chatbot(user_query, conversation_messages, index, metadata, embedding_model, gemini_client):
    """
    Main function to query the chatbot with dual query retrieval.
    
    Flow:
    1. Reformulate query into search-optimized text
    2. Retrieve context using BOTH original + reformulated queries (10 matches total)
    3. Get surrounding chunks (3 before + 3 after) for all matches
    4. Format context with file metadata
    5. Generate answer using all information
    
    Args:
        user_query: The user's question (original)
        conversation_messages: List of previous messages [{"role": "user/assistant", "content": "..."}]
        index: FAISS index
        metadata: Document metadata
        embedding_model: Sentence transformer model
        gemini_client: Gemini API client
    
    Returns:
        Answer string or error message
    """
    try:
        # Step 1: Reformulate query for better retrieval
        reformulated_query = reformulate_query(user_query, conversation_messages, gemini_client)
        
        # Step 2: Retrieve context using BOTH queries (dual retrieval)
        context_chunks = retrieve_context_dual_query(
            user_query,           # Original query
            reformulated_query,   # Reformulated query
            index,
            metadata,
            embedding_model
        )
        
        # Step 3: Build conversation history
        conversation_history = build_conversation_history(conversation_messages)
        
        # Step 4: Generate answer with rich context
        answer = generate_answer(
            user_query,
            reformulated_query,
            context_chunks,
            conversation_history,
            gemini_client
        )
        
        return answer
    
    except Exception as e:
        return f"Error: {str(e)}"


# For standalone testing
if __name__ == "__main__":
    print("Loading models and database...")
    
    try:
        gemini_client = load_gemini_client()
        embedding_model = load_embedding_model()
        index, metadata = load_vector_database()
        
        print("✓ All resources loaded successfully!\n")
        print("="*60)
        print("College Chatbot - Dual Query RAG System")
        print("="*60)
        print("Type 'exit' or 'quit' to end\n")
        
        conversation_messages = []
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Get answer
            print()  # Add spacing
            answer = query_chatbot(
                user_input,
                conversation_messages,
                index,
                metadata,
                embedding_model,
                gemini_client
            )
            
            print(f"Chatbot: {answer}\n")
            
            # Update conversation history
            conversation_messages.append({"role": "user", "content": user_input})
            conversation_messages.append({"role": "assistant", "content": answer})
    
    except Exception as e:
        print(f"Error: {e}")