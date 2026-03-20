import os
import json
import tempfile
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import main as st

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quiz Generator", page_icon="🧠")
st.title("🧠 AI Quiz Generator")
st.caption("Upload your lecture notes and generate a quiz!")

# ─── Session State ────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "questions" not in st.session_state:
    st.session_state.questions = []

if "revealed" not in st.session_state:
    st.session_state.revealed = {}

# ─── Sidebar: PDF Upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Lectures")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

# ─── Process PDFs ─────────────────────────────────────────────────────────────
if process_btn and uploaded_files:
    with st.spinner("Processing your PDFs... ⏳"):

        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)
            os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ""],
            chunk_size=200,
            chunk_overlap=20
        )
        chunks = splitter.split_documents(all_docs)

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory="chroma_db"
        )

        st.session_state.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        st.session_state.questions = []
        st.session_state.revealed = {}

    st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) processed! {len(chunks)} chunks created.")

# ─── LLM + Prompt + Chain ─────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

template = """
You are a quiz generator. Generate {num_questions} questions from the context below.
Question Type: {question_type}
Difficulty: {difficulty}

Context: {context}

Return ONLY a valid JSON object like this:
{{
    "questions": [
        {{
            "type": "MCQ",
            "question": "...",
            "options": ["A", "B", "C", "D"],
            "answer": "..."
        }},
        {{
            "type": "Short",
            "question": "...",
            "answer": "..."
        }}
    ]
}}

Return ONLY JSON. No extra text.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question_type", "difficulty", "num_questions"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = prompt | llm | StrOutputParser()

def generate_quiz(topic, question_type, difficulty, num_questions):
    context = format_docs(st.session_state.retriever.invoke(topic))
    result = chain.invoke({
        "context": context,
        "question_type": question_type,
        "difficulty": difficulty,
        "num_questions": num_questions
    })
    return result

# ─── User Inputs ──────────────────────────────────────────────────────────────
topic = st.text_input("📌 Enter a topic to generate questions about:",
                       placeholder="e.g. Neural Networks, OSI Model, SQL Joins")

question_type = st.selectbox("📝 Question Type:", ["MCQ", "Short Questions", "Both"])

difficulty = st.selectbox("⚡ Difficulty Level:", ["Easy", "Medium", "Hard"])

num_questions = st.slider("🔢 Number of Questions:", min_value=2, max_value=15, value=5)

generate_btn = st.button("🚀 Generate Quiz", use_container_width=True)

# ─── Generate Quiz ────────────────────────────────────────────────────────────
if generate_btn:
    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload and process your PDFs first!")
    elif not topic:
        st.warning("⚠️ Please enter a topic first!")
    else:
        with st.spinner("Generating your quiz... ⏳"):
            try:
                raw = generate_quiz(topic, question_type, difficulty, num_questions)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
                st.session_state.questions = parsed["questions"]
                st.session_state.revealed = {}
            except json.JSONDecodeError:
                st.error("❌ Failed to parse quiz. Try again or rephrase your topic.")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

# ─── Display Questions ────────────────────────────────────────────────────────
if st.session_state.questions:
    st.success(f"✅ {len(st.session_state.questions)} questions generated!")
    st.divider()

    for i, q in enumerate(st.session_state.questions):
        st.subheader(f"Q{i+1}: {q['question']}")

        if q["type"] == "MCQ" and "options" in q:
            for option in q["options"]:
                st.write(f"- {option}")

        if st.button(f"💡 Reveal Answer", key=f"reveal_{i}"):
            st.session_state.revealed[i] = True

        if st.session_state.revealed.get(i):
            st.success(f"✅ Answer: {q['answer']}")

        st.divider()