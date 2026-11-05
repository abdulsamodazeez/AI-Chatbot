# Document Assistant: chat with your PDFs

A retrieval-augmented generation (RAG) app that answers questions about your own PDF documents. Upload one or more PDFs, ask questions in plain language, and get answers drawn only from your documents, with a page citation for every fact.

Typical uses: a support assistant over a product manual, a policy or contract Q&A tool, or a research helper over a stack of reports.

## Features

- **Grounded answers.** The model sees only the passages retrieved from your documents and is instructed to say so when the answer is not there, rather than guess.
- **Page citations.** Every answer cites its sources as `[file.pdf p. 3]`, and a Sources panel shows the exact passages used with their relevance scores.
- **Multiple documents.** Upload several PDFs at once; they are searched together.
- **Follow-up questions.** Recent conversation is kept so you can ask "and what about returns?" without repeating context.
- **Streaming responses.** Answers appear as they are generated.
- **Any OpenAI-compatible provider.** Works with OpenAI by default, or DeepInfra, Together, a local server, or anything else that speaks the OpenAI API.
- **Deploy anywhere.** Docker image (non-root), Procfile for Heroku-style platforms, and CI on every push.

## How it works

```
PDFs ──► extract text per page ──► split into overlapping chunks ──► embed ──► vector index
                                                                                   │
question ──► embed ──► top-k most similar chunks ◄─────────────────────────────────┘
                              │
                              ▼
          system prompt + recent chat + excerpts with page labels ──► LLM ──► streamed answer + sources
```

1. **Extract** (`chatbot/pdf.py`): text is read page by page with `pypdf`; pages with no extractable text are skipped.
2. **Chunk** (`chatbot/chunking.py`): each page is split into overlapping chunks on word boundaries. Chunks never cross pages, so each one maps to a single citable page.
3. **Index** (`chatbot/index.py`): chunks are embedded in batches and stored in an in-memory cosine-similarity index.
4. **Retrieve and answer** (`chatbot/llm.py`): the question is embedded, the top-k chunks are retrieved, and the model answers from those excerpts only.

## Quick start

```bash
git clone https://github.com/abdulsamodazeez/AI-Chatbot.git
cd AI-Chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
streamlit run app.py
```

Open http://localhost:8501, upload a PDF in the sidebar and start asking questions.

### With Docker

```bash
docker build -t document-assistant .
docker run --rm -p 8501:8501 --env-file .env document-assistant
```

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | (required) | API key for your provider |
| `OPENAI_BASE_URL` | OpenAI | Base URL of any OpenAI-compatible API |
| `CHAT_MODEL` | `gpt-4o-mini` | Model that writes the answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model that embeds chunks and questions |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters shared between neighbouring chunks |
| `TOP_K` | `4` | Number of passages retrieved per question |

## Project structure

```
app.py                 Streamlit UI: upload, chat, sources panel
chatbot/
  config.py            Settings from environment variables
  pdf.py               Page-by-page text extraction
  chunking.py          Overlapping, page-bounded chunking
  index.py             In-memory cosine-similarity vector index
  llm.py               Embeddings, grounded prompt, streaming answers
tests/                 Unit and end-to-end tests (no API key needed)
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests build real PDFs in memory and use a deterministic stand-in for the embedding model, so they run offline in under a second. CI runs lint and tests on every push.

## Limitations and next steps

- Scanned PDFs have no text layer and need OCR first.
- The index lives in memory for the session. For large or shared document sets, swap `VectorIndex` for a vector database such as Qdrant or pgvector; the rest of the app does not change.
- Possible additions: OCR for scanned pages, hybrid keyword and vector search, re-ranking, and an evaluation set of questions with known answers to measure retrieval quality.
