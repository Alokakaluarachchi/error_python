import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).parent
HISTORY_PATH = APP_DIR / "data" / "history.json"
SEED_ANSWERS_PATH = APP_DIR / "data" / "seed_answers.json"
SEED_EMBEDDINGS_PATH = APP_DIR / "data" / "seed_embeddings.json"
STACK_API = "https://api.stackexchange.com/2.3/search/advanced"
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="Error Log Helper API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the embedding model so startup is quick.
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_seed_answers() -> List[dict]:
    if not SEED_ANSWERS_PATH.exists():
        return []
    return json.loads(SEED_ANSWERS_PATH.read_text(encoding="utf-8"))


def ensure_seed_embeddings() -> List[dict]:
    """
    Returns seed entries with embeddings. If embedding file is missing,
    it is generated from seed_answers.json and saved to disk.
    """
    if SEED_EMBEDDINGS_PATH.exists():
        return json.loads(SEED_EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    seeds = load_seed_answers()
    if not seeds:
        return []

    model = get_model()
    texts = [f"{s['issue']} {s['answer']}" for s in seeds]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    enriched = []
    for seed, vec in zip(seeds, vectors):
        enriched.append(
            {
                "id": seed.get("id"),
                "tags": seed.get("tags", []),
                "issue": seed.get("issue", ""),
                "answer": seed.get("answer", ""),
                "embedding": vec,
            }
        )

    SEED_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_EMBEDDINGS_PATH.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return enriched


class AnalyzeRequest(BaseModel):
    log: str
    top_k: int = 8


class AnalyzeResponse(BaseModel):
    query: str
    keywords: List[str]
    highlights: List[str]
    results: List[dict]
    local_matches: List[dict] = []


STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "this",
    "that",
    "have",
    "error",
    "exception",
    "stack",
    "trace",
    "line",
    "code",
    "file",
    "module",
    "function",
    "class",
    "failed",
    "fail",
    "cannot",
    "null",
    "undefined",
    "fatal",
    "warning",
    "info",
    "debug",
    "at",
}

ERROR_HINT = np.mean(
    [
        get_model().encode(t, normalize_embeddings=True)
        for t in [
            "error",
            "exception",
            "stack trace",
            "crash",
            "failed",
            "traceback",
            "undefined",
        ]
    ],
    axis=0,
)


def extract_keywords(text: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9._:/-]+", text.lower())
    counts = {}
    for token in tokens:
        if len(token) < 3 or token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


def preprocess_log(log: str) -> Tuple[List[str], str]:
    """
    Returns (highlight_lines, query_string)
    - highlight_lines: the most error-like lines using embedding similarity
    - query_string: concatenated keywords to search
    """
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    if not lines:
        return [], ""

    model = get_model()
    embeddings = model.encode(lines, normalize_embeddings=True)

    sims = embeddings @ ERROR_HINT
    ranked = sorted(zip(lines, sims), key=lambda x: x[1], reverse=True)
    top_lines = [ln for ln, _ in ranked[: min(6, len(ranked))]]

    keywords = extract_keywords(" ".join(top_lines)) or extract_keywords(log)
    query = " ".join(keywords[:6]) or " ".join(top_lines[:3])
    return top_lines, query


def find_local_matches(log: str, seeds: List[dict], top_k: int = 3) -> List[dict]:
    if not seeds:
        return []
    model = get_model()
    log_vec = model.encode(log, normalize_embeddings=True)

    matches = []
    for seed in seeds:
        emb = np.array(seed["embedding"], dtype=np.float32)
        score = float(np.dot(log_vec, emb))
        matches.append(
            {
                "id": seed["id"],
                "tags": seed.get("tags", []),
                "issue": seed.get("issue", ""),
                "answer": seed.get("answer", ""),
                "score": score,
            }
        )

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:top_k]


def search_stackoverflow(query: str, limit: int) -> List[dict]:
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": str(limit),
    }
    res = requests.get(STACK_API, params=params, headers={"Accept": "application/json"}, timeout=12)
    if not res.ok:
        raise HTTPException(status_code=502, detail="Stack Overflow API unavailable")
    data = res.json()
    return data.get("items", [])


def save_history(entry: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        history = []
    history.append(entry)
    # Keep last 50 queries to prevent the file from growing too large.
    HISTORY_PATH.write_text(json.dumps(history[-50:], indent=2), encoding="utf-8")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    log = (payload.log or "").strip()
    if not log:
        raise HTTPException(status_code=400, detail="log is required")

    seeds = ensure_seed_embeddings()
    highlights, query = preprocess_log(log)
    if not query:
        raise HTTPException(status_code=400, detail="could not derive search query")

    results = search_stackoverflow(query, payload.top_k)
    local_matches = find_local_matches(log, seeds)
    entry = {"query": query, "highlights": highlights, "results": len(results), "local_matches": len(local_matches)}
    save_history(entry)
    return AnalyzeResponse(
        query=query,
        keywords=extract_keywords(log),
        highlights=highlights,
        results=results,
        local_matches=local_matches,
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


