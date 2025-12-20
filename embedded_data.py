import os
import glob
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_FOLDER = "data"
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # overlapping words
DB_PATH = "./vector_db"
METADATA_FILE = "metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, efficient local model
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def load_txt_files(data_folder):
    """Load all .txt files from the data folder."""
    txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {data_folder}")
        return []
    
    documents = []
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(file_path)
                documents.append({
                    'content': content,
                    'filename': filename,
                    'filepath': file_path
                })
                print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return documents

def create_embeddings_local(texts, model, batch_size=32):
    """Create embeddings using local sentence-transformers model."""
    print(f"\nGenerating embeddings for {len(texts)} chunks...")
    print("This will be much faster than API calls!\n")
    
    # Encode all texts in batches (much faster than API!)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings

def embed_and_store():
    """Main function to chunk, embed, and store all documents."""
    
    # Create directory for database
    os.makedirs(DB_PATH, exist_ok=True)
    
    # Load local embedding model
    print(f"\nLoading local embedding model: {EMBEDDING_MODEL}")
    print("(First run will download the model, subsequent runs will be instant)\n")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Model: {EMBEDDING_MODEL}")
    print(f"  - Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"  - Max sequence length: {model.max_seq_length} tokens\n")
    
    # Load documents
    print(f"Loading documents from {DATA_FOLDER}...")
    documents = load_txt_files(DATA_FOLDER)
    
    if not documents:
        print("No documents to process. Exiting.")
        return
    
    # Process each document
    all_chunks = []
    all_metadatas = []
    
    for doc in documents:
        filename = doc['filename']
        print(f"\nChunking: {filename}")
        
        chunks = chunk_text(doc['content'])
        total_chunks = len(chunks)
        print(f"Created {total_chunks} chunks from {filename}")
        
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                'filename': filename,
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'text': chunk
            })
    
    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    
    # Create embeddings locally (FAST!)
    print("\n" + "="*60)
    print("Starting local embedding process...")
    print("="*60)
    
    all_embeddings = create_embeddings_local(all_chunks, model)
    
    # Convert to numpy array with correct dtype
    embeddings_array = np.array(all_embeddings).astype('float32')
    
    print(f"\n✓ Generated {len(all_embeddings)} embeddings")
    print(f"  Shape: {embeddings_array.shape}")
    
    # Create FAISS index
    print("\nCreating FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(embeddings_array)
    
    # Save FAISS index
    index_path = os.path.join(DB_PATH, "vector_index.faiss")
    faiss.write_index(index, index_path)
    print(f"✓ FAISS index saved to: {index_path}")
    
    # Save metadata
    metadata_path = os.path.join(DB_PATH, METADATA_FILE)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadatas, f, ensure_ascii=False, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    print(f"\n" + "="*60)
    print(f"✓ Successfully embedded and stored {len(all_chunks)} chunks from {len(documents)} files")
    print("="*60)

if __name__ == "__main__":
    embed_and_store()