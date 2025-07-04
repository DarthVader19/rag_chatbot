import streamlit as st
import os
import requests
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
POST_COMPLAINT_ENDPOINT = f"{API_URL}/complaints"
GET_COMPLAINT_ENDPOINT = f"{API_URL}/complaints/"
KNOWLEDGE_BASE_PATH = "knowledge_base/sample_faq.txt"

# --- RAG Pipeline Setup (Cached) ---
@st.cache_resource
def setup_rag_pipeline(file_path):
    """
    Loads a document, splits it, creates embeddings, and sets up a retriever.
    This function is cached to avoid reloading on every interaction.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()
        
        llm = Ollama(model="llama3.2:3b")
        print("Creating RetrievalQA chain...")
        prompt_template = """
        Use the context to answer the question. If you don't know, say you don't know.
        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Failed to set up RAG pipeline: {e}")
        return None

# --- API Interaction Functions ---
def handle_complaint_creation(details):
    try:
        response = requests.post(POST_COMPLAINT_ENDPOINT, json=details)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

def handle_complaint_retrieval(complaint_id):
    try:
        response = requests.get(f"{GET_COMPLAINT_ENDPOINT}{complaint_id.strip()}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def extract_complaint_id(query):
    match = re.search(r'\b(CMP-[A-Z0-9]{8})\b', query, re.IGNORECASE)
    return match.group(0) if match else None

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Support Chatbot", layout="centered")
st.title("Customer Support Chatbot")
st.write("Ask questions about our services or file a complaint.")

# Initialize RAG chain
rag_chain = setup_rag_pipeline(KNOWLEDGE_BASE_PATH)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
if "collecting_complaint" not in st.session_state:
    st.session_state.collecting_complaint = False
if "complaint_data" not in st.session_state:
    st.session_state.complaint_data = {}

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Chatbot Logic ---
    with st.chat_message("assistant"):
        response = ""
        complaint_id = extract_complaint_id(prompt)

        if complaint_id:
            st.session_state.collecting_complaint = False
            with st.spinner(f"Searching for details for complaint ID: {complaint_id}..."):
                details = handle_complaint_retrieval(complaint_id)
                if details:
                    response = (
                        f"**Complaint Details Found:**\n\n"
                        f"- **ID:** `{details.get('complaint_id')}`\n"
                        f"- **Name:** {details.get('name')}\n"
                        f"- **Phone:** {details.get('phone_number')}\n"
                        f"- **Email:** {details.get('email')}\n"
                        f"- **Details:** {details.get('complaint_details')}\n"
                        f"- **Created At:** {details.get('created_at')}"
                    )
                else:
                    response = f"Sorry, I couldn't find any details for complaint ID `{complaint_id}`."
        
        elif st.session_state.collecting_complaint:
            # Continue collecting complaint details
            if "name" not in st.session_state.complaint_data:
                st.session_state.complaint_data["name"] = prompt
                response = "Thank you. What is your phone number?"
            elif "phone_number" not in st.session_state.complaint_data:
                st.session_state.complaint_data["phone_number"] = prompt
                response = "Got it. What is your email address?"
            elif "email" not in st.session_state.complaint_data:
                st.session_state.complaint_data["email"] = prompt
                
                # All details collected, create the complaint
                with st.spinner("Registering your complaint..."):
                    result = handle_complaint_creation(st.session_state.complaint_data)
                    if "error" in result:
                        response = f"There was an error creating your complaint: {result['error']}"
                    else:
                        response = f"Thank you! Your complaint has been registered. Your Complaint ID is: `{result['complaint_id']}`. You can use this ID to check the status later."
                
                # Reset complaint collection state
                st.session_state.collecting_complaint = False
                st.session_state.complaint_data = {}
        
        elif "complaint" in prompt.lower():
            st.session_state.collecting_complaint = True
            st.session_state.complaint_data = {"complaint_details": prompt}
            response = "I can help with that. To file a complaint, please provide your full name."
            
        else:
            # Use RAG for general queries
            if rag_chain:
                with st.spinner("Thinking..."):
                    rag_response = rag_chain.invoke({"query": prompt})
                    response = rag_response['result']
            else:
                response = "The RAG pipeline is not available. Please check the application setup."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

