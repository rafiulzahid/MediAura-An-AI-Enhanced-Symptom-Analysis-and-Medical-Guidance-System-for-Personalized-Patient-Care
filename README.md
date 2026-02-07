# MediAura: An AI-Enhanced Symptom Analysis and Medical Guidance System for Personalized Patient Care
# AI-Enhanced Medical Guidance System with RAG Chatbot

This project presents an AI-driven healthcare system that integrates disease prediction, personalized medical guidance, a Retrieval-Augmented Generation (RAG) chatbot, and a doctor appointment booking module. The system is designed to support early diagnosis and improve healthcare accessibility, particularly in low- and middle-income countries such as Bangladesh. The project has been developed as a research-oriented system and used in an IEEE-style conference paper.

## Key Features

- Symptom-based disease prediction using machine learning models
- Personalized medical guidance including disease description, precautions, medication, diet, and workout recommendations
- Retrieval-Augmented Generation (RAG) chatbot for grounded medical question answering
- Hybrid document retrieval using semantic search and keyword-based search
- Source-aware and transparent chatbot responses
- Online doctor appointment booking system
- Integration of automated guidance with real doctor consultation

## System Architecture

- Medical Guidance Module
  - Symptom preprocessing and encoding
  - Disease prediction using supervised machine learning
  - Generation of personalized recommendations

- RAG Chatbot Module
  - Medical document ingestion and preprocessing
  - Text chunking and semantic embedding generation
  - Vector storage and similarity search using FAISS
  - Keyword-based retrieval using BM25
  - Ensemble retrieval and cross-encoder reranking
  - Answer generation using FLAN-T5 with source attribution

- Doctor Appointment Module
  - Doctor directory with specialization details
  - Appointment booking interface

## Machine Learning Models

- Random Forest Classifier (selected model, 98 percent accuracy)
- Support Vector Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors
- Naive Bayes

## RAG Chatbot Stack

- Document Loader: PyPDFLoader (LangChain)
- Text Chunking: RecursiveCharacterTextSplitter
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Vector Database: FAISS
- Keyword Retrieval: BM25
- Reranking Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Language Model: google/flan-t5-base
- Orchestration Framework: LangChain

## Dataset

- Medicine Recommendation System Dataset (Kaggle)
- Medical reference documents including The Gale Encyclopedia of Medicine

## Evaluation

- Disease prediction evaluated using accuracy, precision, recall, and F1-score
- Random Forest achieved 98 percent accuracy with balanced precision and recall
- RAG chatbot evaluated using ROUGE-1 and ROUGE-L F1 scores
- Best configuration achieved ROUGE-1 F1 of 0.315 and ROUGE-L F1 of 0.300
- Retrieval performance showed high precision and balanced recall

## How to Run

- Clone the repository
- Install dependencies using requirements.txt
- Run the disease prediction module
- Generate document embeddings and FAISS index
- Run the RAG chatbot module

## Use Cases

- Early disease screening and self-assessment
- AI-assisted medical guidance
- Healthcare support in rural and resource-limited regions
- Medical education and awareness
- Research on RAG-based healthcare systems

## Future Work

- Multimodal AI integration including voice and image inputs
- Multilingual and cross-cultural support
- Explainable AI integration
- Mobile application deployment
- Enhanced personalization using behavioral and lifestyle analytics

## Research Context

- Prepared as part of an IEEE-style conference research paper
- Focused on AI-driven healthcare support for LMICs

## Author

- Rafiul Rabbi Zahid  
- Undergraduate Student, IoT and Robotics Engineering, University of Frontier Technology, Bangladesh
- Bangladesh

## Disclaimer

- This system is intended for educational and research purposes only
- It does not replace professional medical advice
- Users should consult licensed medical professionals for medical decisions
