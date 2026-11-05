import hashlib
import io

import openai
import streamlit as st
from dotenv import load_dotenv

from chatbot.chunking import chunk_pages
from chatbot.config import load_settings
from chatbot.index import SearchResult, VectorIndex
from chatbot.llm import build_messages, make_client, make_embedder, stream_answer
from chatbot.pdf import read_pdf

load_dotenv()
settings = load_settings()

st.set_page_config(page_title="Document Assistant", layout="centered")
st.title("Document Assistant")
st.caption("Ask questions about your PDFs. Answers come only from your documents, with page citations.")

if not settings.api_key:
    st.error("Set OPENAI_API_KEY to start. See .env.example for the available settings.")
    st.stop()


@st.cache_resource
def get_client():
    return make_client(settings)


client = get_client()
embed = make_embedder(client, settings.embedding_model)


def build_index(files: list[tuple[str, bytes]]) -> VectorIndex:
    pages = [page for name, data in files for page in read_pdf(io.BytesIO(data), name)]
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    return VectorIndex.build(chunks, embed)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}, page {s['page']}** · relevance {s['score']:.2f}")
            st.caption(s["text"])


def to_sources(results: list[SearchResult]) -> list[dict]:
    return [
        {"source": r.chunk.source, "page": r.chunk.page, "text": r.chunk.text, "score": r.score}
        for r in results
    ]


with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

if not uploads:
    st.info("Upload one or more PDFs in the sidebar to begin.")
    st.stop()

files = [(f.name, f.getvalue()) for f in uploads]
files_key = hashlib.sha256(b"".join(name.encode() + data for name, data in files)).hexdigest()

if st.session_state.get("files_key") != files_key:
    try:
        with st.spinner("Reading and indexing your documents..."):
            index = build_index(files)
    except openai.OpenAIError as exc:
        st.error(f"Could not index the documents: {exc}")
        st.stop()
    st.session_state.update(index=index, files_key=files_key, messages=[])

index: VectorIndex = st.session_state.index
if not index.chunks:
    st.warning("No extractable text found. Scanned PDFs need OCR before they can be searched.")
    st.stop()

with st.sidebar:
    st.caption(f"{len(files)} file(s) indexed · {len(index.chunks)} passages")

messages: list[dict] = st.session_state.setdefault("messages", [])

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question := st.chat_input("Ask about your documents"):
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    with st.chat_message("assistant"):
        try:
            results = index.search(embed([question])[0], settings.top_k)
            answer = st.write_stream(
                stream_answer(client, settings.chat_model, build_messages(question, results, history))
            )
        except openai.OpenAIError as exc:
            st.error(f"The model request failed: {exc}")
            st.stop()
        sources = to_sources(results)
        render_sources(sources)

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer, "sources": sources})
