from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load the PDF
loader = PyPDFLoader("data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
pages = loader.load()
print(f"✅ PDF loaded. Total pages: {len(pages)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
docs = splitter.split_documents(pages)
print(f"✅ Text chunked. Total chunks: {len(docs)}")

# Use local embedding model (no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector index
db = FAISS.from_documents(docs, embeddings)

# Save the vectorstore to disk
db.save_local("vectorstore")
print("✅ Vectorstore saved to ./vectorstore ✅")
