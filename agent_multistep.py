from dotenv import load_dotenv
from transformers import pipeline
import os
import re




# ---------------------------
# Setup (same as we did before)
# ---------------------------
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise SystemExit("❌ No Hugging Face token found. Add HF_TOKEN=... to your .env")






MODEL_ID = "google/flan-t5-base"

pipe = pipeline(
    task="text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,              # CPU for stability/consistency
)





# ---------------------------
# Tiny helpers (post-process)
# ---------------------------
STOPWORDS = {
    "the","a","an","and","or","for","to","of","in","on","with","at","by","from","is","are",
    "this","that","those","these","about","into","as","it","its","be","being","been","was","were"
}

def one_line(text: str) -> str:
    return text.strip().splitlines()[0].strip()

def clip(text: str, max_chars: int = 160) -> str:
    t = one_line(text)
    return (t[:max_chars].rstrip(" ,.;:!-")) if len(t) > max_chars else t

def title_case_five_words(text: str) -> str:
    # Keep only letters/numbers, split, take 5 non-stopwords if possible.
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return "Short Title Placeholder"
    # prefer non-stopwords first
    content = [w for w in words if w.lower() not in STOPWORDS] or words
    trimmed = content[:5] if len(content) >= 5 else (content + ["Ideas"] * (5 - len(content)))
    return " ".join(w.capitalize() for w in trimmed[:5])

def three_hashtags(summary: str, topic: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", f"{topic} {summary}")
    # prefer meaningful tokens
    candidates = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS or len(wl) < 3:
            continue
        if wl not in seen:
            seen.add(wl)
            candidates.append(wl)
    # fallback if too few
    if len(candidates) < 3:
        candidates.extend(["ai", "learning", "topic"])
    tags = [f"#{candidates[i][:24]}" for i in range(3)]
    return " ".join(tags)





# ---------------------------
# Core multi-step logic
# ---------------------------
def process_topic(topic: str):
    # Step 1: concise summary (deterministic, no sampling)
    p1 = (
        "Write ONE short sentence (<= 18 words) summarizing this topic for a general audience. "
        "Avoid repetition. No list, no title:\n"
        f"Topic: {topic}"
    )
    summary = pipe(p1, do_sample=False, max_new_tokens=40)[0]["generated_text"]
    summary = clip(summary, 140)

    # Step 2: exactly five-word title (deterministic)
    p2 = (
        "Write a catchy title of EXACTLY five words for this summary. "
        "Capitalize Each Word. No punctuation, no quotes:\n"
        f"Summary: {summary}"
    )
    raw_title = pipe(p2, do_sample=False, max_new_tokens=10)[0]["generated_text"]
    title = title_case_five_words(raw_title)

    # If model echoed the summary, enforce five-word title from the summary instead
    if title.lower() in summary.lower() or len(title.split()) != 5:
        title = title_case_five_words(summary)

    # Step 3: three hashtags (deterministic; post-process to ensure format)
    p3 = (
        "Suggest EXACTLY three short hashtags for this topic. "
        "Return ONLY the hashtags separated by single spaces (e.g., #topic #topic #topic). "
        "No extra words:\n"
        f"Summary: {summary}"
    )
    raw_tags = pipe(p3, do_sample=False, max_new_tokens=12)[0]["generated_text"]
    found = re.findall(r"#\w+", raw_tags)
    if len(found) != 3:
        hashtags = three_hashtags(summary, topic)
    else:
        hashtags = " ".join(found[:3])

    return summary, title, hashtags

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("🧠 Multi-Step Agent ready! Type a topic, or 'quit' to stop.\n")
    while True:
        topic = input("Enter a topic: ").strip()
        if topic.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not topic:
            continue

        print("\n--- Thinking in steps ---\n")
        summary, title, hashtags = process_topic(topic)
        print("✅ Final Output:")
        print(f"Summary: {summary}")
        print(f"Title: {title}")
        print(f"Hashtags: {hashtags}\n")