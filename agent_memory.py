# A

import os
import re
from dotenv import load_dotenv
from transformers import pipeline

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise SystemExit("❌ No Hugging Face token found. Add HF_TOKEN=... to your .env")

MODEL_ID = "google/flan-t5-base"  # small demo model

pipe = pipeline(
    task="text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,  # CPU stability
)



# B

# ---------------------------
# Memory stores
# ---------------------------
history: list[str] = []       # chat turns: "You: ...", "Agent: ..."
facts: dict[str, str] = {}    # normalized preferences, e.g., {"sport": "basketball", "food": "ramen"}


# C

# ---------------------------
# Helpers: storing + recalling facts
# ---------------------------
FAV_PAT = re.compile(r"my (?:favorite|best)\s+(\w+)\s+is\s+([A-Za-z ]+)", re.I)
LIKE_PAT = re.compile(r"\bi (?:like|love)\s+([A-Za-z ]+)", re.I)


def normalize_key(word: str) -> str:
    # tiny normalization: plural -> singular for a few common categories
    word = word.lower().strip()
    mapping = {"sports": "sport", "foods": "food", "songs": "song", "movies": "movie"}
    return mapping.get(word, word)

def store_fact(user_message: str):
    """Capture simple ‘favorite X is Y’ and ‘I like/love Y’ facts."""
    m = FAV_PAT.search(user_message)
    if m:
        key = normalize_key(m.group(1))
        val = m.group(2).strip(" .!?,")
        facts[key] = val
        return
    m2 = LIKE_PAT.search(user_message)
    if m2:
        val = m2.group(1).strip()
        facts["general_like"] = val


# D

def answer_from_facts(user_message: str) -> str | None:
    """Answer from stored facts if the question references them."""
    msg = user_message.lower()

    # If message mentions a specific known category, answer that first.
    for key, val in facts.items():
        if key == "general_like":
            continue
        if key in msg:
            return f"You said your favorite {key} is {val}."

    # If user asks about favorites in general and we have exactly one favorite
    if "favorite" in msg and any(k for k in facts if k != "general_like"):
        # Prefer sport if present; otherwise first favorite we have
        if "sport" in facts:
            return f"You said your favorite sport is {facts['sport']}."
        for k, v in facts.items():
            if k != "general_like":
                return f"You said your favorite {k} is {v}."

    # If user asks what they "like/love", try generic like first, else fall back to a favorite
    if "like" in msg or "love" in msg:
        if "general_like" in facts:
            return f"You said you like {facts['general_like']}."
        # If message hints the category (e.g., 'what sport do I like?') use favorite of that category
        for k, v in facts.items():
            if k != "general_like" and k in msg:
                return f"You said you like {v}."
        # Otherwise, if we have exactly one favorite, answer with it
        favorites = [(k, v) for k, v in facts.items() if k != "general_like"]
        if len(favorites) == 1:
            k, v = favorites[0]
            return f"You said your favorite {k} is {v}."

    return None




# E

def oneline(text: str) -> str:
    return text.strip().splitlines()[0].strip()

def hard_sanitize(text: str) -> str:
    """Strip any leaked meta/instruction lines."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Drop lines that look like meta/instructions
        if any(w in s.lower() for w in [
            "rule", "instruction", "do not", "never speak", "context:", "assistant:", "user:", "[", "]", ":", "You are"
        ]):
            continue
        lines.append(s)
    cleaned = " ".join(lines)
    # Secondary cleanup if anything slipped through
    cleaned = re.sub(r"(do not|never speak|instruction).*", "", cleaned, flags=re.I).strip()
    return oneline(cleaned) or "I don't know."


# F

# ---------------------------
# Response logic
# ---------------------------
def respond(user_message: str) -> str:
    # 1) Try symbolic memory first (deterministic, no LLM needed)
    recall = answer_from_facts(user_message)
    if recall:
        return recall

    # 2) Minimal context window (last 6 lines)
    context = "\n".join(history[-6:]) if history else "(no prior messages)"

    # 3) Neutral Q&A style prompt to minimize parroting
    prompt = f"""Answer the user in one short sentence using only the conversation context.

Conversation:
{context}

Question:
{user_message}

Short answer:"""

    raw = pipe(prompt, do_sample=False, max_new_tokens=40)[0]["generated_text"]
    return hard_sanitize(raw)


# G

# ---------------------------
# Run loop
# ---------------------------
if __name__ == "__main__":
    print("🧠 Memory Agent ready! Type anything, or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Save simple facts BEFORE we answer (so recall works immediately)
        store_fact(user_input)

        answer = respond(user_input)

        # Log both sides AFTER we answer (so context reflects the conversation)
        history.append(f"You: {user_input}")
        history.append(f"Agent: {answer}")

        print(f"\nAgent: {answer}\n")