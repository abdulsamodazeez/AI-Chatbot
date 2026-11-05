from types import SimpleNamespace

from chatbot.chunking import Chunk
from chatbot.index import SearchResult
from chatbot.llm import HISTORY_TURNS, SYSTEM_PROMPT, build_messages, make_embedder, stream_answer

RESULTS = [
    SearchResult(Chunk("policy.pdf", 4, "Refunds are processed within 14 days."), 0.91),
    SearchResult(Chunk("policy.pdf", 7, "Store credit never expires."), 0.55),
]


def test_question_is_sent_with_cited_excerpts():
    messages = build_messages("How long do refunds take?", RESULTS, history=[])

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[-1]["content"]
    assert "[policy.pdf p. 4]\nRefunds are processed within 14 days." in user
    assert "[policy.pdf p. 7]" in user
    assert user.endswith("Question: How long do refunds take?")


def test_only_recent_history_is_kept():
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]

    messages = build_messages("next", RESULTS, history)

    assert messages[1:-1] == history[-HISTORY_TURNS:]


def test_no_results_tells_the_model_explicitly():
    messages = build_messages("Anything?", [], history=[])

    assert "(no relevant excerpts found)" in messages[-1]["content"]


def test_stream_answer_yields_only_content():
    def delta(content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    events = [delta("Refunds "), delta(None), SimpleNamespace(choices=[]), delta("take 14 days.")]
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: iter(events)))
    )

    assert "".join(stream_answer(client, "model", [])) == "Refunds take 14 days."


def test_embedder_returns_matrix_in_input_order():
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0]), SimpleNamespace(embedding=[0.0, 1.0])]
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: response))

    vectors = make_embedder(client, "embed-model")(["a", "b"])

    assert vectors.shape == (2, 2)
    assert vectors[1, 1] == 1.0
