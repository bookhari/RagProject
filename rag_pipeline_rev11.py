import os
import pinecone
import openai
import fitz  # PyMuPDF for PDF processing
import time
import arabic_reshaper  # Fixes Arabic letter shapes
from bidi.algorithm import get_display  # Corrects RTL text direction
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve keys and configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
from pinecone import Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Check if the index exists; if not, create it.
index_list = [index.name for index in pc.list_indexes()]
if pinecone_index_name not in index_list:
    pc.create_index(name=pinecone_index_name, dimension=1536)

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF and returns it as a string."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    print("Extracted text preview:", text[:200])
    return text

def process_batch(batch, embeddings):
    """Processes a batch of text chunks with exponential backoff on rate limit errors."""
    success = False
    wait_time = 10  # initial wait time in seconds
    while not success:
        try:
            LangchainPinecone.from_texts(batch, embeddings, index_name=pinecone_index_name)
            success = True
        except Exception as e:
            if "429" in str(e) or "insufficient_quota" in str(e):
                print(f"Rate limit error encountered: {e}")
                print(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                wait_time *= 2  # exponential backoff
            else:
                raise

def embed_and_store_text(pdf_path):
    """Extracts text from a PDF, splits it into chunks, and stores embeddings in Pinecone."""
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    print(f"Number of text chunks: {len(texts)}")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, max_retries=5)
    
    batch_size = 10  # Adjust batch size based on your rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        process_batch(batch, embeddings)
        time.sleep(2)  # Delay between batches

def initialize_rag():
    """Initializes the retrieval chain using the new create_retrieval_chain constructor."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = LangchainPinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)
    retriever = vector_store.as_retriever()
    
    # Arabic prompt template
    prompt = ChatPromptTemplate.from_template("""
    أجب على السؤال بناءً على السياق التالي باللغة العربية:
    {context}
    
    السؤال: {input}
    الإجابة:
    """)
    
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    return qa_chain

def ask_chatgpt_arabic(query):
    """Queries the retrieval chain with an Arabic query and returns the response."""
    qa_chain = initialize_rag()
    # Pass input as a dictionary with "input" key
    response = qa_chain.invoke({"input": query})
    answer = response["answer"]  # Extract the answer from the response dictionary

    # Fix Arabic display order
    reshaped_text = arabic_reshaper.reshape(answer)  # Fix Arabic letter shapes
    bidi_text = get_display(reshaped_text)  # Adjust for proper RTL display
    
    return bidi_text

if __name__ == "__main__":
    pdf_path = "Bronchitis_ARA.pdf"  # Update with the correct file path if necessary
    embed_and_store_text(pdf_path)
    
    user_query = "ما هي اعراض الالتهاب؟"
    answer = ask_chatgpt_arabic(user_query)
    print("الإجابة:", answer)
