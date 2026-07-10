import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq



custom_prompt_template = """
You are a legal assistant. Use the context from the document to answer the user's question.
If the answer is not in the context, say you don't know.

Previous conversation:
{chat_history}

Document context:
{context}

Current question: {question}
Answer:"""


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DB_PATH="vectorstore/db_faiss"


pdfs_directory = 'pdfs/'
llm_model=ChatGroq(model="llama-3.3-70b-versatile")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store(db_faiss_path, text_chunks):
    faiss_db=FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(db_faiss_path)
    return faiss_db


def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


def answer_query(documents, model, query, chat_history=[]):
    context = get_context(documents)
    # Format chat history as readable conversation
    history_text = ""
    for msg in chat_history[-6:]:  # Last 3 exchanges (6 messages)
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context, "chat_history": history_text})


# --- Session state for chat history and FAISS DB ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

# Build vector store when a PDF is uploaded
if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    text_chunks = create_chunks(documents)
    st.session_state.faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks)
    st.success(f"✅ PDF loaded: {uploaded_file.name}")

st.divider()

# Display full chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input at the bottom
user_query = st.chat_input("Ask the AI Lawyer...")

if user_query:
    if st.session_state.faiss_db is None:
        st.error("Please upload a PDF first!")
    else:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Get answer
        retrieved_docs = retrieve_docs(st.session_state.faiss_db, user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query, chat_history=st.session_state.chat_history)
        answer = response.content  # Extract plain text only

        # Show assistant reply
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


