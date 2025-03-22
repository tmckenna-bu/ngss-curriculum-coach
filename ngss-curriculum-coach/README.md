# NGSS Curriculum Coach (Improved)

This is the updated implementation of the NGSS Curriculum Coach, a Retrieval-Augmented Generation (RAG) chatbot designed to support science educators in implementing NGSS-aligned instruction.

## Structure

- `app/` — Core application logic
  - `components/` — Streamlit UI components
  - `utils/` — Response generation and helper utilities
- `data/curriculum/` — Curriculum files (PDFs, DOCX, etc.)
- `data/metadata/` — Metadata file (`curriculum_metadata.json`)
- `data/index/` — Vector store index (e.g., Chroma)
- `prompts/` — Prompt templates for different response types

## Getting Started

1. Clone this repo and install dependencies
2. Place your curriculum files in `data/curriculum/`
3. Update or generate `data/metadata/curriculum_metadata.json`
4. Run the embedding pipeline to generate your vector store
5. Launch the chatbot with Streamlit or CLI interface

