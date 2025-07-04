import os
import requests
import re
# Import new loader for text files
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA

# --- Configuration ---
# API endpoints (No changes here)
API_URL = "http://127.0.0.1:8000"
POST_COMPLAINT_ENDPOINT = f"{API_URL}/complaints"
GET_COMPLAINT_ENDPOINT = f"{API_URL}/complaints/"

# --- 1. RAG System Setup (MODIFIED) ---
def setup_rag_pipeline(file_path):
    """
    Loads a document (PDF or TXT), splits it, creates embeddings, 
    and sets up a retriever.
    """
    # --- MODIFIED PART: Select loader based on file extension ---
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt are supported.")

    print(f"Loading documents from {file_path}...")
    documents = loader.load()
    
    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)

    # Set up the retriever
    retriever = vector_store.as_retriever()
    
    # Set up the LLM to use Ollama
    llm = Ollama(model="llama3.2:3b")

    # Create the RetrievalQA chain
    prompt_template = """
    Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
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

# --- 2. Complaint Management & API Integration (No changes here) ---
def handle_complaint_creation(details):
    """Calls the API to create a complaint."""
    try:
        response = requests.post(POST_COMPLAINT_ENDPOINT, json=details)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

def handle_complaint_retrieval(complaint_id):
    """Calls the API to get complaint details."""
    try:
        response = requests.get(f"{GET_COMPLAINT_ENDPOINT}{complaint_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def extract_complaint_id(query):
    """Extracts a complaint ID from the user's query using regex."""
    match = re.search(r'\b(CMP-[A-Z0-9]{8})\b', query, re.IGNORECASE)
    return match.group(0) if match else None

# --- 3. Main Chatbot Logic (MODIFIED) ---
def main():
    """Main function to run the chatbot interaction loop."""
    print("Setting up the RAG pipeline with Ollama...")
    
    # --- MODIFIED PART: Point to the new .txt file ---
    # You can change this back to your .pdf file if you want
    knowledge_base_path = os.path.join("..", "knowledge_base", "sample_faq.txt")
    
    if not os.path.exists(knowledge_base_path):
        print(f"Error: Knowledge base file not found at {knowledge_base_path}")
        return

    rag_chain = setup_rag_pipeline(knowledge_base_path)
    print("Chatbot is ready! Make sure Ollama is running. Type 'quit' to exit.")

    complaint_data = {}
    collecting_complaint = False

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        complaint_id = extract_complaint_id(user_input)

        if complaint_id:
            # Handle complaint retrieval
            print(f"Chatbot: Searching for details for complaint ID: {complaint_id}...")
            details = handle_complaint_retrieval(complaint_id)
            if details:
                response = (
                    f"Complaint ID: {details.get('complaint_id')}\n"
                    f"Name: {details.get('name')}\n"
                    f"Phone: {details.get('phone_number')}\n"
                    f"Email: {details.get('email')}\n"
                    f"Details: {details.get('complaint_details')}\n"
                    f"Created At: {details.get('created_at')}"
                )
                print(f"Chatbot:\n{response}")
            else:
                print("Chatbot: Sorry, I couldn't find any details for that complaint ID.")
            continue
            
        if "complaint" in user_input.lower() and not collecting_complaint:
            collecting_complaint = True
            complaint_data = {"complaint_details": user_input}
            print("Chatbot: I can help with that. To file a complaint, I need some details.")

        if collecting_complaint:
            # Maintain conversation context to collect details
            if "name" not in complaint_data:
                complaint_data["name"] = input("Chatbot: Please provide your full name: ")
            elif "phone_number" not in complaint_data:
                complaint_data["phone_number"] = input("Chatbot: What is your phone number? ")
            elif "email" not in complaint_data:
                complaint_data["email"] = input("Chatbot: And your email address, please? ")
            else:
                # All details collected, create the complaint
                print("Chatbot: Thank you. Registering your complaint...")
                result = handle_complaint_creation(complaint_data)
                
                if "error" in result:
                    print(f"Chatbot: There was an error: {result['error']}")
                else:
                    print(f"Chatbot: Your complaint has been registered with ID: {result['complaint_id']}. You'll hear back soon.")
                
                # Reset for next interaction
                collecting_complaint = False
                complaint_data = {}
        else:
            # Use the RAG chain for general queries
            response = rag_chain.invoke({"query": user_input})
            print(f"Chatbot: {response['result']}")

if __name__ == "__main__":
    main()
