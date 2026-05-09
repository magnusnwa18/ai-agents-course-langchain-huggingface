from dotenv import load_dotenv
from transformers import pipeline
import os

# 1) Load your token (added in 01_01)
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise SystemExit("❌ No Hugging Face token found. Add HF_TOKEN=... to your .env")





# 2) Pick a small model that runs fine on CPU
MODEL_ID = "google/flan-t5-base"

# 3) Build a text-generation pipeline (CPU mode to avoid Mac MPS issues)
pipe = pipeline(
    task="text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,  # CPU for consistency and stability on Mac
)






# 4) A tiny prompt template: 1 line, setup — punchline
TEMPLATE = (
    "Write one clean, original one-line joke about {topic}. "
    "Format: Setup — Punchline. Keep it under 18 words."
)



def make_joke(topic="computers"):
    prompt = TEMPLATE.format(topic=topic)
    result = pipe(
        prompt,
        do_sample=True,      # small dose of variety
        top_p=0.92,
        top_k=50,
        max_new_tokens=40
    )[0]["generated_text"]
    # keep the first line if model adds extras
    return result.splitlines()[0].strip()





if __name__ == "__main__":
    print("🤖 Joke-Teller ready! Type a topic, or 'quit' to stop.\n")
    while True:
        topic = input("Topic: ").strip()
        if topic.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not topic:
            topic = "computers"
        print(f"\n😂 {make_joke(topic)}\n")