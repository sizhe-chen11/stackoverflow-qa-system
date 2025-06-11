# Stack Overflow QA System

A RAG-based question-answering system using ChatGLM and Stack Overflow data.

## Features

- 🤖 ChatGLM-6B for natural language generation
- 🔍 Semantic Search using sentence transformers  
- 📚 Document Retrieval with Chroma vector database
- 💬 Interactive CLI for real-time Q&A

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sizhe-chen11/stackoverflow-qa-system.git
cd stackoverflow-qa-system

# Install dependencies  
conda create -n qa-system python=3.10 -y
conda activate qa-system
pip install -r requirements.txt

# Run
python main.py
