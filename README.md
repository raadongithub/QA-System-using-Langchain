# ASR Question-Answering Application using Langchain & RAG

## Overview

This repository contains an AI-powered Question-Answering application that leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based on uploaded documents. The application uses Google's Gemini AI model and LangChain for document processing and retrieval.

## Contents

- `app.py`: Main application script
- `requirements.txt`: List of Python dependencies

## Features

- Supports multiple document formats: PDF, DOCX, and TXT
- Utilizes Google's Gemini AI for natural language understanding and generation
- Implements RAG for enhanced answer accuracy
- Interactive chat interface using Streamlit
- Document chunking and embedding for efficient retrieval

## Installation

1. Clone this repository:
- git clone https://github.com/raadongithub/QA-System-using-Langchain.git 
- cd QA-System-using-Langchain

2. Install dependencies:
`pip install -r requirements.txt`

3. Create a `.env` file in the project root directory and add your Google API key:

4. ## Usage

1. Run the Streamlit app:

2. 2. Open the provided URL in your web browser.

3. Enter your Google API key in the sidebar.

4. Upload a document (PDF, DOCX, or TXT) using the file uploader.

5. Ask questions about the document in the chat interface.

## How It Works

1. **Document Processing**: 
- The application accepts PDF, DOCX, or TXT files.
- Documents are loaded using appropriate libraries (PyPDF2 for PDF, python-docx for DOCX).
- Text is extracted and split into smaller chunks using LangChain's RecursiveCharacterTextSplitter.

2. **Embedding and Indexing**:
- Text chunks are embedded using HuggingFace's sentence-transformers model.
- Embeddings are stored in a FAISS vector store for efficient similarity search.

3. **Question-Answering**:
- User questions trigger a search for relevant document chunks.
- Retrieved chunks and the question are sent to the Gemini AI model.
- The model generates a response based on the provided context and question.

4. **Chat Interface**:
- Streamlit and streamlit-chat are used to create an interactive chat experience.
- Chat history is maintained for context in follow-up questions.

## Technologies Used

- [Streamlit](https://streamlit.io/): For creating the web interface
- [LangChain](https://python.langchain.com/): For document processing, embedding, and retrieval
- [Google Generative AI (Gemini)](https://ai.google.dev/): For natural language understanding and generation
- [FAISS](https://github.com/facebookresearch/faiss): For efficient similarity search of embeddings
- [HuggingFace Transformers](https://huggingface.co/): For text embedding
- [PyPDF2](https://pypdf2.readthedocs.io/): For PDF processing
- [python-docx](https://python-docx.readthedocs.io/): For DOCX processing

## Retrieval-Augmented Generation (RAG)

This application implements RAG, a technique that enhances large language models with external knowledge. RAG combines the power of retrieval-based systems with generative models:

1. The retriever finds relevant information from the uploaded document.
2. The generator (Gemini AI) uses this information to produce more accurate and contextually relevant answers.

This approach allows the model to access specific information from the document without needing to encode all details in its parameters, resulting in more accurate and up-to-date responses.

## Future Improvements

- Support for more document formats
- Integration with additional AI models
- Enhanced error handling and user feedback
- Optimization of chunking and retrieval strategies
- Implementation of source citation in responses

## Contributing

Contributions to improve the application are welcome. Please feel free to submit issues and pull requests.
