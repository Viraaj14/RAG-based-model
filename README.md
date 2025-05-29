# 🧠 Hybrid RAG + LLM Chatbot

This project implements a hybrid chatbot that intelligently decides between:
- **RAG (Retrieval-Augmented Generation)**: to answer questions based on a set of local `.txt` documents.
- **LLM-only Generation**: for general-purpose or creative queries without needing external context.

Built using:
- 🤗 Hugging Face Transformers
- 🦜 LangChain + FAISS for document retrieval
- 🧬 SentenceTransformers for embeddings
- ⚡ Mistral-7B-Instruct (quantized for efficiency)

---

## 📁 Project Structure

.
├── main.py # Core chatbot logic (RAG + LLM)
├── documents/ # Folder containing .txt documents
├── README.md # This file

yaml
Copy
Edit

---

## 🛠️ Setup Instructions

### 1. ✅ Install dependencies

Use pip to install required packages:

```bash
pip install transformers torch accelerate sentence-transformers langchain langchain-huggingface faiss-cpu
If you're using a GPU, consider using faiss-gpu instead of faiss-cpu.

2. 📄 Prepare Your Documents
Create a folder named documents and place your .txt files inside. These will be used as the knowledge base for RAG-based responses.

3. 🚀 Run the Chatbot
Launch the chatbot using:
python main.py

Type your question when prompted. To quit, type:
exit

💡 Features
🔍 RAG Mode
Retrieves relevant document chunks using semantic similarity.

Answers questions only using retrieved context.

🧠 LLM Mode
Handles general NLP tasks like summarization, translation, rewriting, etc.

Uses recent chat history for conversational continuity.

🤖 Smart Routing Logic
Decides between RAG and LLM using:

Keyword detection (e.g., "summarize", "translate")

Semantic similarity scores
