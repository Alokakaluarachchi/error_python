import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
KB_FILE = 'knowledge_base.json'
MODEL_NAME = 'all-MiniLM-L6-v2'  # Excellent, fast, and resource-efficient model
K_RESULTS = 2 # Number of similar errors to return

# --- Method 1: Extraction Function ---

def extract_core_error(log_text):
    """
    Uses regex and keyword filtering to extract the core error snippet.
    Focuses on lines starting with common error indicators.
    """
    
    # 1. Look for the error marker (ERROR, EXCEPTION, FATAL, etc.)
    # We will try to capture the error line plus up to 10 subsequent lines of the stack trace.
    # The pattern is: (Start of line) (Error keyword) (Capture the rest of the line and next 10 lines)
    # The (?:.|\n){1,10} captures up to 10 more lines of context.
    pattern = r"^(?:ERROR|EXCEPTION|FATAL|java\.\w+\.\w+Exception|Timeout|Permission denied).{0,300}(?:.|\n){0,10}"
    
    # Use re.MULTILINE for ^ to match the start of lines, and re.IGNORECASE
    match = re.search(pattern, log_text, re.MULTILINE | re.IGNORECASE)
    
    if match:
        # Clean up the captured snippet (remove leading/trailing whitespace)
        snippet = match.group(0).strip()
        
        # Further refinement: Remove verbose timestamps/thread IDs from the start of the snippet if present
        # This regex removes common log prefixes (like [2025-12-08 13:00:00 INFO [Thread-5]])
        snippet = re.sub(r"^\s*\[\d{4}-\d{2}-\d{2}.+?\]\s*(?:INFO|DEBUG|WARN)?\s*", "", snippet, flags=re.MULTILINE)
        
        return snippet
    else:
        # Fallback: If no clear error is found, return the first few lines 
        # (useful for logs that are entirely an error message without a prefix)
        return log_text.split('\n', 2)[0] if log_text else ""


# --- 2. Indexing Phase (Setup & Model Loading) ---

def load_knowledge_base(file_path):
    """Loads the knowledge base from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def build_index(knowledge_base, model):
    """Generates embeddings and builds the FAISS index."""
    texts_to_embed = [
        f"{entry['core_log_snippet']} {entry['resolution_summary']}"
        for entry in knowledge_base
    ]
    
    print(f"Embedding {len(texts_to_embed)} knowledge base entries...")
    embeddings = model.encode(texts_to_embed, convert_to_numpy=True)
    d = embeddings.shape[1]
    
    # Using IndexFlatL2 for fast L2 distance calculation
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

# --- 3. Retrieval Phase (Search Logic) ---

def find_similar_errors(raw_log, model, index, knowledge_base, k):
    """Embeds the query and performs a FAISS search."""
    
    # STEP 1: EXTRACT CORE ERROR PART
    query_error = extract_core_error(raw_log)
    print(f"\n--- Searching with Extracted Snippet: '{query_error.splitlines()[0]}...' ---")
    
    # STEP 2: EMBED THE CLEANED QUERY
    query_vector = model.encode([query_error], convert_to_numpy=True)
    
    # STEP 3: PERFORM SEARCH
    D, I = index.search(query_vector, k)
    
    results = []
    for i in range(k):
        kb_index = I[0][i]
        distance = D[0][i]
        original_entry = knowledge_base[kb_index]
        
        results.append({
            "rank": i + 1,
            "id": original_entry['id'],
            "similarity_score (L2 Distance)": round(distance, 4), 
            "error_type": original_entry['error_type'],
            "original_log_snippet": original_entry['core_log_snippet'],
            "resolution": original_entry['resolution_summary']
        })
        
    return results

# --- Main Execution ---

if __name__ == "__main__":
    # Load Model and KB Data
    print("Loading Sentence Transformer Model...")
    model = SentenceTransformer(MODEL_NAME)
    knowledge_base_data = load_knowledge_base(KB_FILE)
    faiss_index = build_index(knowledge_base_data, model)

    # --- SIMULATED HUGE LOGS ---
    
    # Test Log 1: Contains verbose logs before and after the error
    huge_log_1 = """
    [2025-12-08 13:00:01 INFO [Thread-3]] Starting background process for user 1234.
    [2025-12-08 13:00:02 DEBUG [Thread-3]] Attempting to get connection from pool with timeout 3000ms.
    [2025-12-08 13:00:03 ERROR [Thread-3]] Timeout expired. The timeout period elapsed prior to obtaining a connection from the pool.
        at System.Data.ProviderBase.DbConnectionPool.GetConnection()
        at System.Data.SqlClient.SqlConnection.Open()
        at App.Data.DatabaseHelper.ExecuteQuery() in DatabaseHelper.cs:line 45
    [2025-12-08 13:00:04 INFO [Thread-4]] Process 1234 failed. Attempting cleanup.
    """

    # Test Log 2: Contains a Java NPE error in the middle of log lines
    huge_log_2 = """
    [2025-12-08 14:00:10 INFO [HTTP-A]] Processing request for /api/v2/data. Auth token received.
    [2025-12-08 14:00:11 DEBUG [HTTP-A]] Token validation in progress.
    [2025-12-08 14:00:12 EXCEPTION [HTTP-A]] java.lang.NullPointerException: Cannot read field 'user' because 'auth_token' is null
        at com.app.UserService.loadUserData(UserService.java:88)
        at com.app.Controller.handleRequest(Controller.java:52)
    [2025-12-08 14:00:13 WARN [HTTP-A]] Response sent with status 500.
    """

    # --- DEMO EXECUTION ---
    
    print("\n============== RUNNING TEST CASE 1 (DB TIMEOUT) ==============")
    similar_errors_1 = find_similar_errors(huge_log_1, model, faiss_index, knowledge_base_data, K_RESULTS)
    
    for result in similar_errors_1:
        print(f"----------------------------------------")
        print(f"Rank {result['rank']} (Score: {result['similarity_score (L2 Distance)']}) | ID: {result['id']} ({result['error_type']})")
        print(f"-> **KB Snippet:** {result['original_log_snippet']}")
        print(f"-> **Resolution:** {result['resolution']}")

    print("\n============== RUNNING TEST CASE 2 (NPE) ==============")
    similar_errors_2 = find_similar_errors(huge_log_2, model, faiss_index, knowledge_base_data, K_RESULTS)
    
    for result in similar_errors_2:
        print(f"----------------------------------------")
        print(f"Rank {result['rank']} (Score: {result['similarity_score (L2 Distance)']}) | ID: {result['id']} ({result['error_type']})")
        print(f"-> **KB Snippet:** {result['original_log_snippet']}")
        print(f"-> **Resolution:** {result['resolution']}")
        
    print("\nDemo complete. The log cleaning step is now integrated.")
