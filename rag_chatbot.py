"""A simple RAG chatbot.

1. Load .txt / .md files from ./docs and split them into chunks.
2. Embed each chunk with a local Ollama embedding model and store it in a ChromaDB
   vector database (saved in ./vector_db, rebuilt automatically when ./docs changes).
3. For each question, retrieve the most relevant chunks. RETRIEVAL picks the method:
   "vector" (meaning), "bm25" (keywords), or "hybrid" (both, merged).
4. Send the chunks plus the question to an LLM and stream the answer.

Setup and usage: see README.md.

Two backends (PROVIDER setting):
  - "ollama": free, runs locally. No key needed.
  - "claude": Claude API. Needs ANTHROPIC_API_KEY.

Every setting below can be overridden with an environment variable of the same
name (e.g. OLLAMA_MODEL=qwen2.5:7b), so nothing needs editing on a new machine.
"""

import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

PROVIDER = os.getenv("PROVIDER", "ollama")  # "ollama" or "claude"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

RETRIEVAL = os.getenv("RETRIEVAL", "hybrid")  # "vector", "bm25", or "hybrid"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "docs"))
DB_DIR = Path(os.getenv("DB_DIR", BASE_DIR / "vector_db"))
CHUNK_WORDS = 200
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context excerpts. "
    "Base your answer on the context. If the context doesn't contain the answer, say you don't know "
    "rather than guessing. Mention the source file you used in brackets, taken from the excerpt's source attribute."
)


# ---------- Indexing ----------

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its me my "
    "of on or our should so that the their there this to was we what when where which "
    "who will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def load_chunks(docs_dir: Path) -> list[dict]:
    """Split files into chunks along headings and paragraphs, so tables and
    sections stay whole. Each chunk is prefixed with its section heading."""
    chunks = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        heading, buf = "", []

        def flush():
            text = "\n\n".join(buf).strip()
            if text:
                prefix = f"[Section: {heading}]\n" if heading else ""
                chunks.append({"source": path.name, "text": prefix + text})
            buf.clear()

        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            block = block.strip()
            if not block or block == "---":
                continue
            if block.startswith("#"):
                flush()
                lines = block.splitlines()
                heading = lines[0].lstrip("#").strip()
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            if buf and len(" ".join(buf).split()) + len(block.split()) > CHUNK_WORDS:
                flush()
            buf.append(block)
        flush()
    return chunks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_freqs = [Counter(t) for t in self.doc_tokens]
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(len(docs), 1)
        df = Counter(term for tokens in self.doc_tokens for term in set(tokens))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scores = []
        for i, freqs in enumerate(self.doc_freqs):
            dl = len(self.doc_tokens[i])
            score = 0.0
            for term in terms:
                if term not in freqs:
                    continue
                tf = freqs[term]
                score += self.idf[term] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda s: s[1], reverse=True)[:k]


def ollama_post(endpoint: str, payload: dict):
    req = urllib.request.Request(
        OLLAMA_HOST + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama error: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Can't reach Ollama at {OLLAMA_HOST} - is the Ollama app running? ({e})")


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    """Turn texts into vectors. nomic-embed-text expects a task prefix on each text."""
    vectors = []
    for start in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[start:start + 32]]
        with ollama_post("/api/embed", {"model": EMBED_MODEL, "input": batch}) as resp:
            vectors.extend(json.load(resp)["embeddings"])
    return vectors


class VectorStore:
    """Chunk embeddings stored in a local ChromaDB collection."""

    def __init__(self, chunks: list[dict]):
        import chromadb

        client = chromadb.PersistentClient(path=str(DB_DIR))
        # Fingerprint of the docs + embedding model; re-embed only when it changes.
        fingerprint = hashlib.sha256(
            json.dumps([EMBED_MODEL, chunks], sort_keys=True).encode()
        ).hexdigest()
        name = "docs"
        try:
            self.collection = client.get_collection(name)
            if (self.collection.metadata or {}).get("fingerprint") == fingerprint:
                print(f"Loaded {self.collection.count()} vectors from {DB_DIR}.")
                return
            client.delete_collection(name)
        except Exception:
            pass  # collection doesn't exist yet

        print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
        self.collection = client.create_collection(
            name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"fingerprint": fingerprint},
            embedding_function=None,  # we supply our own vectors
        )
        texts = [c["text"] for c in chunks]
        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embed(texts, "search_document: "),
            documents=texts,
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        print(f"Saved vectors to {DB_DIR}.")

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        result = self.collection.query(
            query_embeddings=embed([query], "search_query: "), n_results=k
        )
        # Chroma returns cosine distance; similarity = 1 - distance.
        return [(int(i), 1 - d) for i, d in zip(result["ids"][0], result["distances"][0])]


def hybrid_search(vectors: VectorStore, bm25: BM25, query: str, k: int) -> list[int]:
    """Merge vector and keyword rankings with reciprocal rank fusion."""
    scores: Counter = Counter()
    for results in (vectors.search(query, k * 2), bm25.search(query, k * 2)):
        for rank, (i, _) in enumerate(results):
            scores[i] += 1 / (60 + rank)
    return [i for i, _ in scores.most_common(k)]


# ---------- Chat ----------

def build_user_message(question: str, hits: list[dict]) -> str:
    if not hits:
        context = "(no relevant excerpts found)"
    else:
        context = "\n\n".join(
            f'<excerpt source="{h["source"]}">\n{h["text"]}\n</excerpt>' for h in hits
        )
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


def stream_ollama(history: list[dict], result: dict):
    """Yield answer text from a local Ollama model as it arrives.
    When done, result["message"] holds the assistant message for the history."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "stream": True,
    }
    answer = []
    with ollama_post("/api/chat", payload) as resp:
        for line in resp:
            part = json.loads(line)
            if "error" in part:
                raise RuntimeError(part["error"])
            text = part.get("message", {}).get("content", "")
            answer.append(text)
            yield text
            if part.get("done"):
                break
    result["message"] = {"role": "assistant", "content": "".join(answer)}


def stream_claude(client, history: list[dict], result: dict):
    """Yield answer text from Claude as it arrives.
    When done, result["message"] holds the assistant message for the history."""
    import anthropic

    try:
        with client.beta.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=history,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # retry on a fallback model if the request is declined
        ) as stream:
            yield from stream.text_stream
            response = stream.get_final_message()
    except anthropic.APIError as e:
        raise RuntimeError(f"API error: {e}")
    if response.stop_reason == "refusal":
        yield "\n[The request was declined.]"
    # Keep the full content (including thinking blocks) so the conversation stays valid.
    result["message"] = {"role": "assistant", "content": response.content}


class RagBot:
    """Loads the documents, builds the search indexes, and answers questions.
    Used by both the terminal chat (this file) and the web app (app.py)."""

    def __init__(self):
        self.chunks = load_chunks(DOCS_DIR)
        if not self.chunks:
            raise RuntimeError(f"No .txt or .md files found in {DOCS_DIR}")
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.vectors = VectorStore(self.chunks) if RETRIEVAL in {"vector", "hybrid"} else None
        self.client = None
        if PROVIDER == "claude":
            import anthropic

            self.client = anthropic.Anthropic()
        self.model = OLLAMA_MODEL if PROVIDER == "ollama" else CLAUDE_MODEL

    def retrieve(self, question: str) -> list[dict]:
        if RETRIEVAL == "vector":
            ids = [i for i, _ in self.vectors.search(question, TOP_K)]
        elif RETRIEVAL == "hybrid":
            ids = hybrid_search(self.vectors, self.bm25, question, TOP_K)
        else:
            ids = [i for i, _ in self.bm25.search(question, TOP_K)]
        return [self.chunks[i] for i in ids]

    def stream_answer(self, history: list[dict], result: dict):
        if PROVIDER == "ollama":
            return stream_ollama(history, result)
        return stream_claude(self.client, history, result)


def check_setup() -> None:
    """Fail early with a clear message if something this machine needs is missing."""
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")
    if PROVIDER not in {"ollama", "claude"}:
        sys.exit(f'PROVIDER must be "ollama" or "claude", not "{PROVIDER}".')
    if RETRIEVAL not in {"vector", "bm25", "hybrid"}:
        sys.exit(f'RETRIEVAL must be "vector", "bm25" or "hybrid", not "{RETRIEVAL}".')

    needed = []
    if PROVIDER == "ollama":
        needed.append(OLLAMA_MODEL)
    if RETRIEVAL in {"vector", "hybrid"}:
        needed.append(EMBED_MODEL)
        try:
            import chromadb  # noqa: F401
        except ImportError:
            sys.exit("ChromaDB isn't installed. Run: pip install -r requirements.txt")
    if PROVIDER == "claude":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            sys.exit("The anthropic package isn't installed. Run: pip install -r requirements.txt")

    if needed:
        try:
            with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as resp:
                installed = {m["name"] for m in json.load(resp)["models"]}
        except (urllib.error.URLError, OSError):
            sys.exit(
                f"Can't reach Ollama at {OLLAMA_HOST}.\n"
                "Install it from https://ollama.com and make sure it's running."
            )
        for model in needed:
            if model not in installed and f"{model}:latest" not in installed:
                sys.exit(f"Ollama model '{model}' isn't downloaded. Run: ollama pull {model}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # so symbols like the rupee sign print on Windows
    check_setup()
    try:
        bot = RagBot()
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"{len(bot.chunks)} chunks from {DOCS_DIR} | retrieval: {RETRIEVAL} | model: {bot.model}")
    print("Ask a question (Ctrl+C or 'exit' to quit).\n")

    history: list[dict] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            hits = bot.retrieve(question)
        except RuntimeError as e:
            print(f"[{e}]\n")
            continue
        history.append({"role": "user", "content": build_user_message(question, hits)})

        print("Bot: ", end="", flush=True)
        result: dict = {}
        try:
            for text in bot.stream_answer(history, result):
                print(text, end="", flush=True)
        except RuntimeError as e:
            print(f"\n[{e}]\n")
            history.pop()
            continue

        print()
        sources = sorted({h["source"] for h in hits})
        if sources:
            print(f"   (retrieved from: {', '.join(sources)})")
        print()
        history.append(result["message"])


if __name__ == "__main__":
    main()
