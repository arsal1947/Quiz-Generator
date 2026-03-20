# AI Quiz Generator

An AI-powered quiz generator that turns your PDF lecture notes into interactive quizzes using RAG (Retrieval-Augmented Generation), LangChain, Groq, and Streamlit.

---

## Overview

This app allows students to upload their lecture PDFs and automatically generate quizzes on any topic within those documents. It retrieves the most relevant content using semantic search and passes it to **LLaMA 3.3 70B** via Groq to produce structured quiz questions in JSON format.

---

## Features

- 📄 Upload multiple PDF lecture files
- 🔍 Semantic search using HuggingFace embeddings + ChromaDB
- 🤖 AI-generated questions via LLaMA 3.3 70B (Groq)
- 📝 Supports MCQ, Short Questions, or Both
- ⚡ Choose difficulty: Easy, Medium, or Hard
- 🔢 Select number of questions (2–15)
- 💡 Answers hidden by default, revealed on click

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Framework | LangChain (LCEL) |
| PDF Loader | PyPDFLoader |

---


---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/arsal1947/Quiz-Generator.git
cd Quiz-Generator
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🚀 How to Use

1. Upload one or more PDF lecture files from the sidebar
2. Click **⚡ Process PDFs** to index the content
3. Enter a topic (e.g., `Neural Networks`, `OSI Model`)
4. Choose question type, difficulty, and number of questions
5. Click **🚀 Generate Quiz**
6. Click **💡 Reveal Answer** to see answers one by one

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-groq
langchain-core
langchain-text-splitters
pypdf
chromadb
sentence-transformers
python-dotenv
```

---

## 🔐 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key for LLaMA 3.3 70B access |

---

## 👤 Author

**Arsal** — AI/ML Student  
GitHub: [@arsal1947](https://github.com/arsal1947)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
