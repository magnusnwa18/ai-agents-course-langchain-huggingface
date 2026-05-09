
# agent_tools.py
# Tool-Enabled Agent with robust search:
# - Safe Calculator (pure Python)
# - Web Search: ddgs -> Wikipedia REST -> offline facts -> LLM one-liner
# - FLAN-T5 fallback for general text

import os
import re
import ast
import operator as op
import requests
from dotenv import load_dotenv
from transformers import pipeline
from ddgs import DDGS

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("⚠️ No HF token found in .env. Public/cached models may still work.")

MODEL_ID = "google/flan-t5-base"
llm = pipeline(
    task="text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,  # CPU
)

# ---------------------------
# Tool 1: Safe Calculator
# ---------------------------
SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}

def _eval_ast(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        fn = SAFE_OPS.get(type(node.op))
        if not fn:
            raise ValueError("Unsupported operator")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Expr):
        return _eval_ast(node.value)
    raise ValueError("Unsupported expression")

def safe_calculate(expr: str) -> str:
    try:
        expr = expr.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        return f"{_eval_ast(tree.body)}"
    except Exception:
        return "I couldn't compute that safely."

# ---------------------------
# Tool 2: Web Search (layered fallbacks)
# ---------------------------
def canon(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s]", " ", q)   # drop punctuation like ?!'’,.
    q = re.sub(r"\s+", " ", q)           # collapse spaces
    return q

OFFLINE_FACTS = {
    "capital of japan": "Tokyo",
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "who founded hugging face": "Clément Delangue, Julien Chaumond, and Thomas Wolf.",
    "what is langchain": "LangChain is a framework for building LLM-powered apps using chains, tools, and agents.",
    "define agentic ai": "Agentic AI refers to AI systems that can plan, choose tools, and act autonomously toward goals.",
}

# 1

def _ddgs_search(query: str) -> str | None:
    try:
        with DDGS() as ddg:
            results = list(ddg.text(
                keywords=query,
                max_results=3,
                safesearch="moderate",
                region="us-en"
            ))
        if not results:
            return None
        top = results[0]
        title = (top.get("title") or "").strip()
        snippet = (top.get("body") or "").strip()
        return f"{title} — {snippet}".strip(" —")
    except Exception:
        return None
# 2

def _wiki_summary(query: str) -> str | None:
    try:
        # very light heuristic: if asking for capitals, extract the place
        ql = canon(query)
        term = query
        if ql.startswith("capital of "):
            term = query.lower().replace("capital of ", "").strip(" ?!.,")

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term.title().replace(' ', '%20')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if "extract" in data and data["extract"]:
                return data["extract"].split(". ")[0]
        return None
    except Exception:
        return None
# 3

def _offline_answer(query: str) -> str | None:
    key = canon(query)
    return OFFLINE_FACTS.get(key)
# 4

def web_search(query: str) -> str:
    # 1) ddgs
    hit = _ddgs_search(query)
    if hit:
        return hit
    # 2) Wikipedia summary
    hit = _wiki_summary(query)
    if hit:
        return hit
    # 3) Offline facts (normalized)
    hit = _offline_answer(query)
    if hit:
        return hit
    # 4) Last-resort LLM one-liner definition/explanation
    prompt = f"Answer in one short, factual sentence suitable for a beginner: {query}"
    out = llm(prompt, do_sample=False, max_new_tokens=40)[0]["generated_text"].strip()
    return out.splitlines()[0].strip() or "No results found."

# 5

# ---------------------------
# Simple router
# ---------------------------
def is_math(q: str) -> bool:
    return any(ch.isdigit() for ch in q) and any(sym in q for sym in "+-*/%()^")

def should_search(q: str) -> bool:
    ql = q.lower()
    return any(kw in ql for kw in [
        "who", "when", "where", "what is", "what's", "latest", "capital",
        "define", "search", "find", "founder", "founded", "meaning of"
    ])

def agent_reply(user_input: str) -> str:
    if is_math(user_input):
        return f"Calculator: {safe_calculate(user_input)}"
    if should_search(user_input):
        return f"Search: {web_search(user_input)}"
    # LLM fallback for general chit-chat
    prompt = f"Answer in one short sentence: {user_input}"
    out = llm(prompt, do_sample=False, max_new_tokens=40)[0]["generated_text"].strip()
    return out.splitlines()[0].strip()

# 6

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("⚙️ Tool-Enabled Agent ready! Type a question or 'quit' to stop.\n")
    while True:
        q = input("> ").strip()
        if q.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not q:
            continue
        print("\n🤖", agent_reply(q), "\n")