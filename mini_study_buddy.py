# mini_study_buddy.py
# Simple final project: prompt + multi-step + tiny memory + one tool (calculator)

import os, re, ast, operator as op
from dotenv import load_dotenv
from transformers import pipeline

# ---------------------------
# Setup (same pattern as earlier)
# ---------------------------
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("⚠️ No HF token found in .env. Public/cached models may still work.")

MODEL_ID = "google/flan-t5-base"  # CPU-friendly
llm = pipeline(
    task="text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,
)

def llm_once(prompt: str, max_new_tokens: int = 48) -> str:
    try:
        out = llm(prompt, do_sample=False, max_new_tokens=max_new_tokens)[0].get("generated_text", "").strip()
    except Exception:
        out = ""
    return out.splitlines()[0].strip() if out else ""


# ---------------------------
# Tiny memory (facts) + prefs
# ---------------------------
facts = {}          # e.g., {"topic":"databases"}
tone  = "neutral"   # "neutral" | "playful" | "formal"

TONE_RULE = {
    "neutral": "Use clear, plain wording.",
    "playful": "Add light, friendly energy (no slang).",
    "formal":  "Use professional, concise language.",
}

def set_tone(cmd: str):
    global tone
    m = re.match(r"\s*tone\s+is\s+(neutral|playful|formal)\s*$", cmd, re.I)
    if m:
        tone = m.group(1).lower()
        return True
    return False

def set_topic(cmd: str):
    m = re.match(r"\s*remember\s+my\s+topic\s+is\s+(.+)$", cmd, re.I)
    if m:
        facts["topic"] = m.group(1).strip().rstrip(".!")
        return True
    return False

# ---------------------------
# One tool: Safe calculator
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
        return SAFE_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Expr):
        return _eval_ast(node.value)
    raise ValueError("Unsupported expression")

def is_math(q: str) -> bool:
    return any(ch.isdigit() for ch in q) and any(sym in q for sym in "+-*/%()^")

def calculate(expr: str) -> str:
    try:
        expr = expr.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        return str(_eval_ast(tree.body))
    except Exception:
        return "I couldn't compute that safely."

# ---------------------------
# Multi-step: Title + 2 bullets
# ---------------------------
def make_title(text: str) -> str:
    rule = TONE_RULE[tone]
    p = (
        "Write a short, catchy title (5–7 words). "
        f"Tone: {tone}. {rule} Return only the title (no quotes/punctuation).\n\nText:\n{text[:500]}"
    )
    title = llm_once(p, 16)
    # Fallback: build from first sentence if the model is weak
    words = title.split()
    if len(words) < 3:
        first = _sentences(text[:200])[0] if _sentences(text[:200]) else text[:60]
        words = first.split()
    title = " ".join(words[:7]).rstrip(".,;:!?")
    return title if title else "Quick Summary"



def _sentences(text: str) -> list[str]:
    # Very simple sentence split (no extra deps)
    parts = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    return parts

def _compress_sentence(s: str, max_words: int = 18) -> str:
    words = s.split()
    if not words:
        return "(key point)."
    clipped = " ".join(words[:max_words]).rstrip(",;:—- ")
    if not clipped.endswith("."):
        clipped += "."
    return clipped

def two_bullets(text: str) -> list[str]:
    rule = TONE_RULE[tone]
    p = (
        "Summarize the text as EXACTLY two bullet points. "
        "Each bullet must be a complete sentence under 18 words.\n"
        f"Tone: {tone}. {rule}\n\nText:\n{text}\n\n"
        "Return bullets starting with '• ' on separate lines."
    )
    out = llm_once(p, 96)

    # Try to parse LLM bullets (only if we actually got text back)
    bullets = []
    if out:
        for ln in out.splitlines():
            if ln.strip().startswith("•"):
                btxt = ln.strip().lstrip("• ").strip()
                bullets.append("• " + _compress_sentence(btxt, 18))

    # Fallback if we didn't get two usable bullets
    if len(bullets) < 2:
        sents = _sentences(text)
        if sents:
            for s in sents[:2]:
                bullets.append("• " + _compress_sentence(s, 18))
        if len(bullets) < 2:
            bullets.append("• Key benefit: it reduces costs or effort.")
            bullets.append("• Key impact: it strengthens community or outcomes.")

    return bullets[:2]



# ---------------------------
# Router
# ---------------------------
def respond(user: str) -> str:
    # Preferences / memory commands
    if set_tone(user):
        return f"Preferences updated → tone={tone}"
    if set_topic(user):
        return f"Noted. I’ll keep '{facts['topic']}' in mind."

    # Calculator
    if is_math(user):
        return f"Calculator: {calculate(user)}"

  
    # Content mode: use remembered topic if user asks
    text = user
    if user.lower().strip() == "use my topic":
        if "topic" not in facts or not facts["topic"].strip():
            return "No topic saved yet. Type: remember my topic is <your topic>"
        text = facts["topic"]


    # Multi-step
    title = make_title(text)
    bullets = two_bullets(text)
    return "===== One-Pager =====\n" + \
           f"Title: {title}\n\n" + \
           "Summary:\n" + "\n".join(bullets)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("🎯 Mini Study Buddy ready!")
    print("- Type ‘tone is playful’ | ‘tone is formal’ | ‘tone is neutral’")
    print("- Type ‘remember my topic is <text>’")
    print("- Type a math expression like 50*65")
    print("- Paste any text, or type ‘use my topic’\n")

    while True:
        q = input("> ").strip()
        if q.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not q:
            continue
        print("\n" + respond(q) + "\n")
