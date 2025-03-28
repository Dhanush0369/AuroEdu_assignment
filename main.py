# Connect to the PostgreSQL database and define the database table.

# Implement the add_document API to add documents.

# Generate vector embeddings for the documents and store them in the database.

# Implement the select_documents API to allow users to specify document preferences.

# Implement the ask_question API to handle user queries.


# All the APIs are tested and running perfectly
# There is a more efficient method where we don’t need to store embeddings manually, PGVector can automate the process
# However, due to time constraints, I was unable to implement it
 

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Initialize FastAPI
app = FastAPI()
load_dotenv()


# PostgreSQL Database URL
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base Model
Base = declarative_base()

# Define Documents Table
class DocumentDB(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False)  
    content = Column(Text, nullable=False)  
    embedding = Column(Vector(384)) 

# Create Table in Database
Base.metadata.create_all(bind=engine)

# Dependency to Get DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Root Route
@app.get("/")
def home():
    return {"message": "RAG Backend is running! 🚀"}

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Function to Split Text into Chunks
def split_text(extracted_data):
    return text_splitter.split_documents(extracted_data) 

# Pydantic Model for Request
class DocumentRequest(BaseModel):
    name: str
    content: str

# API to Add a Document with Chunking
@app.post("/add_document/")
def add_document(doc: DocumentRequest, db: Session = Depends(get_db)):
    try:
        extracted_data = [Document(page_content=doc.content)]

        chunks = split_text(extracted_data)

        for chunk in chunks:
            embedding_vector = embedding_model.embed_query(chunk.page_content)  # Extract text from Document object

            new_doc = DocumentDB(name=doc.name, content=chunk.page_content, embedding=embedding_vector)
            db.add(new_doc)
        
        db.commit() 
        
        return {"message": "Document added successfully!", "chunks_stored": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Store selected chunk IDs (in-memory)
selected_document_ids = []

@app.post("/select_documents/")
def select_documents(doc_name: str, db: Session = Depends(get_db)):
    """
    Allows users to select a document by name.
    Retrieves all chunk IDs related to the document.
    """
    global selected_document_ids

    # Get all chunk IDs for the given document name
    chunk_ids = db.query(DocumentDB.id).filter(DocumentDB.name == doc_name).all()
    chunk_ids = [id[0] for id in chunk_ids]  # Convert list of tuples to list

    if not chunk_ids:
        raise HTTPException(status_code=404, detail="Incorrect document name")

    selected_document_ids = chunk_ids

    return {"message": "Document selected successfully!", "selected_ids": selected_document_ids}


# Set Hugging Face credentials
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature= 0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )
    return llm


llm = load_llm(HUGGINGFACE_REPO_ID)

class query(BaseModel):
    ques: str


@app.post("/ask/")
def ask_question(query: query, db: Session = Depends(get_db)):
    """
    Retrieves relevant document chunks using similarity search and generates an answer.
    """
    global selected_document_ids

    # Check if documents have been selected
    if not selected_document_ids:
        raise HTTPException(status_code=400, detail="No documents selected. Please select a document first.")

    selected_chunks = db.query(DocumentDB.content, DocumentDB.embedding).filter(DocumentDB.id.in_(selected_document_ids)).all()

    if not selected_chunks:
        raise HTTPException(status_code=404, detail="No embeddings found for selected documents.")


    # Load embeddings into pgvector for similarity search
    vectorstore = PGVector(
        collection_name="documents",
        connection_string=DATABASE_URL,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    answer = qa_chain.run(query.ques)

    return {"query": query.ques, "answer": answer}