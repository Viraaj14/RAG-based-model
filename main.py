from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
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
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)
    return db

# === Step 4: Load LLM Model (Mistral or compatible) ===
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300, do_sample=False, temperature=0.7, top_k=50, top_p=0.95)
    return pipe

# === Step 5: Decide if RAG is needed ===
def needs_retrieval(query, vector_db):
    general_keywords = [
        "how many words", "summarize", "translate", "rephrase",
        "make it shorter", "what's the meaning", "write a tweet",
        "convert this", "grammar", "fix", "paraphrase"
    ]

    # Direct detection for common general language tasks
    if any(kw in query.lower() for kw in general_keywords):
        return False

    # If the question is factual or about your dataset — trigger retrieval
    domain_keywords = [
        "MPD", "Lahaina", "officer", "evacuation", "fire", "report",
        "August 8", "roadblock", "incident", "dispatcher", "how many injured"
    ]
    return any(kw in query.lower() for kw in domain_keywords)

# def needs_retrieval(query):
#     keywords = ["when", "where", "according to", "report", "how many", "what is", "who", "based on"]
#     return any(kw in query.lower() for kw in keywords)

# === Step 6: Generate Answer (RAG or General) ===
def generate_answer(query, llm, vector_db=None):
    if needs_retrieval(query, vector_db):
        # === RAG Mode ===
        docs = vector_db.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content.strip() for doc in docs])
        prompt = (
            f"[INST] Use the following context to answer the question accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer: [/INST]"
        )
    else:
        # === General LLM Mode ===
        prompt = f"[INST] {query} [/INST]"

    # Use the LLM pipeline to generate only the final answer
    result = llm(prompt)[0]['generated_text']

    # Strip everything before the final answer (only keep what's after [/INST])
    if "[/INST]" in result:
        result = result.split("[/INST]")[-1].strip()

    return result

# === Step 7: Run Chatbot ===
def run_chat():
    print("🔧 Loading documents...")
    raw_docs = load_documents("dataset")  # <-- put your path here
    chunks = chunk_documents(raw_docs)
    db = embed_documents(chunks)
    print("📚 Documents loaded and indexed.")

    print("🤖 Loading language model...")
    llm = load_llm()
    print("✅ Chatbot is ready!")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = generate_answer(user_input, llm, db)
        print("\nBot:", answer)

# Run the chatbot
if __name__ == "__main__":
    run_chat()
