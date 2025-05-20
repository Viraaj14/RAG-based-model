from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

# === Step 1: Load and Read Text Files ===
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(page_content=text))
    return docs

# === Step 2: Split into Chunks ===
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# === Step 3: Embed Chunks ===
def embed_documents(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(chunks, embedding_model)
    return db

# === Step 4: Load LLM Model (Mistral or OpenChat) ===
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

    # 🔍 Concise: for RAG
    rag_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False  # deterministic
    )

    # 🎨 Creative: for LLM-only generation
    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=True,       # allow randomness
        temperature=0.8,      # creative
        top_k=40,
        top_p=0.95
    )

    return rag_pipe, llm_pipe


# === Step 5: Decide if RAG is needed ===
def needs_retrieval(query, vector_db, threshold=0.65):
    general_keywords = [
        "how many words", "summarize", "translate", "rephrase",
        "make it shorter", "what's the meaning", "write a tweet",
        "convert this", "grammar", "fix", "paraphrase", "simplify",
        "improve", "edit", "shorten", "proofread", "spell check"
    ]

    query_lower = query.lower()

    if any(kw in query_lower for kw in general_keywords):
        print(f"[Routing] General task matched → LLM only")
        return False

    try:
        results = vector_db.similarity_search_with_score(query, k=1)
        top_score = results[0][1]
        print(f"[Routing] Semantic similarity score: {top_score:.3f}")
        if top_score < threshold:
            print("[Routing] Matched → Using RAG")
        else:
            print("[Routing] Not similar enough → Using LLM only")
        return top_score < threshold

    except Exception as e:
        print(f"[Routing ERROR]: {e}")
        return True  # fallback to RAG

# === Step 6: Generate Answer (RAG or General) ===
def generate_answer(query, rag_llm, creative_llm, vector_db=None):
    mode_used = "LLM"

    if needs_retrieval(query, vector_db):
        # === RAG Mode ===
        docs = vector_db.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content.strip() for doc in docs])
        prompt = (
            f"[INST] Answer the following question using ONLY the information from the provided context. "
            f"If the answer is not in the context, say 'I don't know.' Do not guess.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer: [/INST]"
        )
        model_to_use = rag_llm
        mode_used = "RAG"
    else:
        # === LLM Mode ===
        prompt = f"[INST] {query} [/INST]"
        model_to_use = creative_llm

    result = model_to_use(prompt)[0]['generated_text']

    # Clean output
    if "[/INST]" in result:
        result = result.split("[/INST]")[-1].strip()

    return result, mode_used

# === Step 7: Run Chatbot ===
def run_chat():
    print("🔧 Loading documents...")
    raw_docs = load_documents("hawai_1")  # Change path as needed
    chunks = chunk_documents(raw_docs)
    db = embed_documents(chunks)
    print("📚 Documents loaded and indexed.")

    print("🤖 Loading language model...")
    rag_llm, creative_llm = load_llm()
    print("✅ Chatbot is ready!")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer, mode = generate_answer(user_input, rag_llm, creative_llm, db)

        print(f"\n[MODE USED]: {mode}")
        print(f"Bot ({mode}): {answer}")

# Run the chatbot
if __name__ == "__main__":
    run_chat()


# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# import os

# # === Step 1: Load and Read Text Files ===
# def load_documents(folder_path):
#     docs = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
#                 text = f.read()
#                 docs.append(Document(page_content=text))
#     return docs

# # === Step 2: Split into Chunks ===
# def chunk_documents(documents):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     return splitter.split_documents(documents)

# # === Step 3: Embed Chunks ===
# def embed_documents(chunks):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.from_documents(chunks, embedding_model)
#     return db

# # === Step 4: Load LLM Model (Mistral or compatible) ===
# def load_llm():
#     model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300, do_sample=False, temperature=0.7, top_k=50, top_p=0.95)
#     return pipe

# # === Step 5: Decide if RAG is needed ===
# def needs_retrieval(query, vector_db, threshold=0.35):
#     """
#     Smart hybrid logic to decide if document context is needed.
#     - Step 1: If query matches known general task keywords, skip RAG
#     - Step 2: If it doesn't, test semantic similarity to the documents
#     - Step 3: If similarity is low, treat it as general too (safety fallback)
#     """
#     general_keywords = [
#         "how many words", "summarize", "translate", "rephrase",
#         "make it shorter", "what's the meaning", "write a tweet",
#         "convert this", "grammar", "fix", "paraphrase", "simplify",
#         "improve", "edit", "shorten", "proofread", "spell check"
#     ]

#     query_lower = query.lower()

#     # Step 1: Keyword rule
#     if any(kw in query_lower for kw in general_keywords):
#         return False

#     try:
#         # Step 2: Semantic match to document data
#         results = vector_db.similarity_search_with_score(query, k=1)
#         top_score = results[0][1]

#         # Step 3: If the query isn't close to any doc content → skip RAG
#         return top_score < threshold

#     except Exception as e:
#         print(f"⚠️ Retrieval decision failed: {e}")
#         return True  # Fallback: use RAG just in case

# # def needs_retrieval(query):
# #     keywords = ["when", "where", "according to", "report", "how many", "what is", "who", "based on"]
# #     return any(kw in query.lower() for kw in keywords)

# # === Step 6: Generate Answer (RAG or General) ===
# def generate_answer(query, llm, vector_db=None):
#     mode_used = "LLM" 

#     if needs_retrieval(query, vector_db):
#         # === RAG Mode ===
#         docs = vector_db.similarity_search(query, k=4)
#         context = "\n\n".join([doc.page_content.strip() for doc in docs])
#         prompt = (
#             f"[INST] Use the following context to answer the question accurately.\n\n"
#             f"Context:\n{context}\n\n"
#             f"Question: {query}\nAnswer: [/INST]"
#         )
#     else:
#         # === General LLM Mode ===
#         prompt = f"[INST] {query} [/INST]"

#     # Use the LLM pipeline to generate only the final answer
#     result = llm(prompt)[0]['generated_text']

#     # Strip everything before the final answer (only keep what's after [/INST])
#     if "[/INST]" in result:
#         result = result.split("[/INST]")[-1].strip()

#     return result, mode_used

# # === Step 7: Run Chatbot ===
# def run_chat():
#     print("🔧 Loading documents...")
#     raw_docs = load_documents("dataset")  # <-- put your path here
#     chunks = chunk_documents(raw_docs)
#     db = embed_documents(chunks)
#     print("📚 Documents loaded and indexed.")

#     print("🤖 Loading language model...")
#     llm = load_llm()
#     print("✅ Chatbot is ready!")

#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
#         answer, mode = generate_answer(user_input, llm, db)
#         print(f"\nBot ({mode}): {answer}")

# # Run the chatbot
# if __name__ == "__main__":
#     run_chat()
