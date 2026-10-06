"""Web chat page for the RAG chatbot, built with Gradio.

    python app.py            # open http://localhost:7860
    python app.py --share    # also create a temporary public link to show other people
"""

import argparse
import sys

import gradio as gr

from rag_chatbot import DOCS_DIR, RETRIEVAL, RagBot, build_user_message, check_setup

EXAMPLES = [
    "What documents do I need to open a savings account?",
    "Someone stole my debit card. What should I do?",
    "What is the FD interest rate for senior citizens?",
    "How do I complain to the RBI Ombudsman?",
]


def text_of(content) -> str:
    """Gradio may store message content as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def to_llm_history(history: list[dict]) -> list[dict]:
    """Previous turns as plain text, skipping the 'retrieved excerpts' panels."""
    messages = []
    for m in history:
        if (m.get("metadata") or {}).get("title"):
            continue
        text = text_of(m.get("content"))
        if text and m.get("role") in {"user", "assistant"}:
            messages.append({"role": m["role"], "content": text})
    return messages


def make_respond(bot: RagBot):
    def respond(message: str, history: list[dict]):
        question = text_of(message).strip()
        if not question:
            return
        try:
            hits = bot.retrieve(question)
        except RuntimeError as e:
            yield f"⚠️ {e}"
            return

        excerpts = "\n\n---\n\n".join(f"**{h['source']}**\n\n{h['text']}" for h in hits)
        context_panel = gr.ChatMessage(
            role="assistant",
            content=excerpts or "No relevant excerpts found.",
            metadata={
                "title": f"📚 Retrieved {len(hits)} excerpts ({RETRIEVAL} search)",
                "status": "done",  # shows the panel collapsed; click to open
            },
        )

        llm_history = to_llm_history(history)
        llm_history.append({"role": "user", "content": build_user_message(question, hits)})

        answer = ""
        try:
            for text in bot.stream_answer(llm_history, {}):
                answer += text
                yield [context_panel, gr.ChatMessage(role="assistant", content=answer)]
        except RuntimeError as e:
            answer += f"\n\n⚠️ {e}"
        sources = ", ".join(sorted({h["source"] for h in hits}))
        if sources:
            answer += f"\n\n<sub>Sources: {sources}</sub>"
        yield [context_panel, gr.ChatMessage(role="assistant", content=answer)]

    return respond


def main() -> None:
    parser = argparse.ArgumentParser(description="Web chat page for the RAG chatbot.")
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    check_setup()
    try:
        bot = RagBot()
    except RuntimeError as e:
        sys.exit(str(e))

    docs = ", ".join(sorted({c["source"] for c in bot.chunks}))
    demo = gr.ChatInterface(
        make_respond(bot),
        title="📄 RAG Chatbot",
        description=(
            f"Ask questions about the documents in `{DOCS_DIR.name}/` ({docs}). "
            f"Answers come from **{bot.model}**, using **{RETRIEVAL}** search over a ChromaDB vector database. "
            "Open **📚 Retrieved excerpts** to see what the bot read before answering."
        ),
        examples=EXAMPLES,
        run_examples_on_click=True,
        cache_examples=False,
        flagging_mode="never",
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
