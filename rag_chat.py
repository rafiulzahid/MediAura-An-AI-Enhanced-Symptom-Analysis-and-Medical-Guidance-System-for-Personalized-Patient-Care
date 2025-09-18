# rag_chat.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Retrieval upgrades
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------- Load vectorstore ----------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

# Vector retriever with MMR (diverse & relevant)
faiss_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.8},
)

# BM25 lexical retriever (helps keyword-heavy medical queries)
# Grab all docs out of the FAISS docstore for BM25
all_docs = list(getattr(db.docstore, "_dict", {}).values())  # fallback for FAISS docstore
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 8

# Combine vector + BM25
ensemble = EnsembleRetriever(retrievers=[faiss_retriever, bm25], weights=[0.65, 0.35])

# Cross-encoder reranker to keep only the best chunks
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# Final retriever = ensemble + reranker
retriever = ContextualCompressionRetriever(
    base_retriever=ensemble,
    base_compressor=reranker,
)

# ---------- Load instruction-following local model ----------
model_id = "google/flan-t5-base"   # stays CPU-friendly; upgrade to flan-t5-large if you have RAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

gen_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
)
llm = HuggingFacePipeline(pipeline=gen_pipe)

# ---------- Grounded prompt ----------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a careful medical assistant. Answer ONLY using the context below. "
        "If the answer is not clearly present, reply: \"I don't know based on the provided document.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Give a concise clinical answer (tests, criteria, steps). No outside knowledge."
    ),
)

# ---------- Build QA chain ----------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True,
)

def get_answer(query: str):
    """Used by Flask."""
    result = qa.invoke(query)
    answer = result["result"]
    srcs = []
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source", "PDF")
        page = doc.metadata.get("page")
        srcs.append(f"{source}" + (f" (p.{page})" if page is not None else ""))
    return {"answer": answer, "sources": srcs}

# CLI test
if __name__ == "__main__":
    while True:
        q = input("\nAsk your medical question (or 'exit'): ")
        if q.lower() == "exit":
            break
        out = get_answer(q)
        print("\n🤖 Answer:\n", out["answer"])
        if out["sources"]:
            print("\n📄 Sources:")
            for s in out["sources"]:
                print("-", s)
