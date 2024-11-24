import streamlit as st
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Vectorstore DB
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit App Config
st.set_page_config(page_title="Document Q&A with OpenAI", layout="wide")

# App Title
st.title("📄 Document Q&A with OpenAI")
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload your documents into the `./PDF` folder (supports PDF, TXT, or DOCX).
    2. Click "Generate Embeddings" to prepare the database.
    3. Ask questions based on the uploaded documents.
    """
)

# Load OpenAI API Key
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    st.error("Error: OPENAI_API_KEY is not set. Please check your .env file.")

# Initialize the language model
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.3)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate and concise response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create vector embeddings
def create_vector_embeddings():
    try:
        # Initialize embeddings and document loader
        st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
        st.session_state.loader = PyPDFDirectoryLoader("./PDF")
        st.session_state.docs = st.session_state.loader.load()
        
        # Document Preview Option
        if st.button("Preview Loaded Documents"):
            st.write("### Preview of Loaded Documents:")
            for doc in st.session_state.docs[:5]:
                st.write(doc.page_content[:500])
                st.write("----")
        
        # Split documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB successfully created!")
    except Exception as e:
        st.error(f"Error creating vector embeddings: {e}")

# Save and Load Vector DB
def save_vector_db():
    try:
        st.session_state.vectors.save_local("./faiss_vectors")
        st.success("Vector DB saved successfully!")
    except Exception as e:
        st.error(f"Error saving vector DB: {e}")

def load_vector_db():
    try:
        st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
        st.session_state.vectors = FAISS.load_local("./faiss_vectors", st.session_state.embeddings)
        st.success("Vector DB loaded successfully!")
    except Exception as e:
        st.error(f"Error loading vector DB: {e}")

# Q&A Functionality
def answer_question(question):
    try:
        # Check if vector DB exists
        if "vectors" not in st.session_state:
            st.error("Vector DB not found. Please create embeddings first.")
            return
        
        # Create retrieval and question-answering chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        st.write("Response time:", round(time.process_time() - start, 2), "seconds")
        
        # Display the answer
        st.markdown("### Answer:")
        st.write(response['answer'])
        
        # Show similarity search results
        with st.expander("🔍 Document Similarity Search Results"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Document {i + 1}: {doc.page_content}")
                st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error processing question: {e}")

# Sidebar Options
st.sidebar.subheader("Vector DB Management")
if st.sidebar.button("Generate Embeddings"):
    create_vector_embeddings()

if st.sidebar.button("Save Vector DB"):
    save_vector_db()

if st.sidebar.button("Load Vector DB"):
    load_vector_db()

# Main App Input
prompt_input = st.text_input("Enter your question based on the documents:")

# Process Q&A
if st.button("Get Answer"):
    if prompt_input:
        answer_question(prompt_input)
    else:
        st.error("Please enter a question.")
