import streamlit as st
import os
import shutil
import time
import subprocess
import sys
import whisper
from yt_dlp import YoutubeDL
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq # New Import

# --- UI Configuration ---
st.set_page_config(page_title="NoteMind AI", page_icon="🧠", layout="wide")
st.title("🧠 NoteMind AI: Your Personal Knowledge Base")

# --- Conditional LLM Initialization ---
# Use Groq if an API key is available in secrets, otherwise use local Ollama
if 'GROQ_API_KEY' in st.secrets:
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)
    st.sidebar.success("✅ Using Groq API")
else:
    llm = OllamaLLM(model="llama3:8b")
    st.sidebar.info("💡 Using local Ollama model")

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("tiny")
    return model

whisper_model = load_whisper_model()

# --- Session State Initialization ---
if 'qa_chain' not in st.session_state: st.session_state.qa_chain = None
if 'vector_db' not in st.session_state: st.session_state.vector_db = None

# --- Helper Functions ---
def initialize_qa_chain(vector_db):
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain

def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if st.session_state.vector_db is not None:
        st.session_state.vector_db.add_documents(chunks)
        st.info("New documents added.")
    else:
        st.session_state.vector_db = Chroma.from_documents(
            documents=chunks, embedding=embeddings
        )
        st.info("New knowledge base created.")
    
    st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_db)
    st.success("Documents processed successfully!")

def transcribe_youtube_audio(url):
    audio_file = None
    try:
        temp_folder = "temp_audio"
        if not os.path.exists(temp_folder): os.makedirs(temp_folder)

        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': os.path.join(temp_folder, '%(id)s.%(ext)s'),
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]}

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_id = info_dict.get("id")
            audio_file = os.path.join(temp_folder, f"{video_id}.mp3")
        
        if not os.path.exists(audio_file):
            st.error("Audio download failed.")
            return None
        
        st.info("Audio downloaded. Starting transcription...")
        result = whisper_model.transcribe(audio_file, fp16=False)
        st.info("Transcription complete!")
        transcript_text = result['text']
        
        document = Document(page_content=transcript_text, metadata={"source": url})
        return [document]

    except Exception as e:
        st.error(f"Failed to process YouTube URL: {e}")
        return None
    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

# --- Sidebar ---
with st.sidebar:
    st.header("Manage Knowledge Base")
    uploaded_files = st.file_uploader("Upload documents for this session", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True, key="file_uploader")

    st.header("Add from URL")
    url_input = st.text_input("Enter a web page URL:", key="url_input_widget")
    if st.button("Process Web Page"):
        if url_input: st.session_state.url_to_process = url_input
        else: st.warning("Please enter a URL.")

    youtube_url_input = st.text_input("Enter a YouTube URL:", key="youtube_url_widget")
    if st.button("Process YouTube Video"):
        if youtube_url_input: st.session_state.youtube_url_to_process = youtube_url_input
        else: st.warning("Please enter a YouTube URL.")
            
    if st.button("Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Processing Logic ---
if uploaded_files:
    with st.spinner("Processing files..."):
        all_docs = []
        if not os.path.exists("temp"): os.makedirs("temp")
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join("temp", uploaded_file.name)
            with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            file_extension = os.path.splitext(uploaded_file.name)[1]
            loader = None
            if file_extension == ".pdf": loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".txt": loader = TextLoader(temp_file_path)
            elif file_extension == ".docx": loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".md": loader = UnstructuredMarkdownLoader(temp_file_path)
            
            if loader: all_docs.extend(loader.load())
            os.remove(temp_file_path)
        
        process_documents(all_docs)

if st.session_state.get("url_to_process"):
    with st.spinner("Fetching and processing URL..."):
        url_to_process = st.session_state.url_to_process
        del st.session_state.url_to_process
        loader = WebBaseLoader(url_to_process)
        documents = loader.load()
        process_documents(documents)
        
if st.session_state.get("youtube_url_to_process"):
    with st.spinner("Downloading audio from YouTube..."):
        url_to_process = st.session_state.youtube_url_to_process
        del st.session_state.youtube_url_to_process
        
        documents = transcribe_youtube_audio(url_to_process)
        
        if documents:
            process_documents(documents)

# --- No loading from disk in this version ---

st.write("---")
st.header("Ask Questions About Your Knowledge Base")

if st.session_state.qa_chain:
    user_question = st.text_input("What would you like to know?")
    if user_question:
        with st.spinner("Searching for the answer..."):
            response = st.session_state.qa_chain.invoke({"query": user_question})
            st.write("### Answer")
            st.write(response["result"])
            st.write("### Sources")
            for source in response["source_documents']:
                st.info(f"Source: {source.metadata.get('source', 'N/A')}")
else:
    st.warning("Please upload documents or add a URL to begin a session.")