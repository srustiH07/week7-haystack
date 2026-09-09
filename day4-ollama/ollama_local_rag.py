import requests
import time


OLLAMA_URL = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer questions clearly and accurately.
Keep answers simple and concise.
"""


def ask_ollama(model, question):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    elapsed = time.perf_counter() - start_time

    result = response.json()

    answer = result["message"]["content"]

    return answer, elapsed


# ============================================================
# Task 2 - Test 5 prompts using llama3.2:3b
# ============================================================

prompts = [
    "What is artificial intelligence?",
    "Explain machine learning in simple words.",
    "What is RAG and why is it useful?",
    "What is a vector database?",
    "Explain large language models in simple terms."
]


print("=" * 70)
print("W7D4 - OLLAMA LOCAL LLM INFERENCE")
print("=" * 70)

print("\nModel: llama3.2:3b")
print("System prompt configured successfully.")


for number, prompt in enumerate(prompts, start=1):

    print("\n" + "-" * 70)
    print(f"Prompt {number}: {prompt}")
    print("-" * 70)

    try:
        answer, elapsed = ask_ollama(
            "llama3.2:3b",
            prompt
        )

        print("Response:")
        print(answer)

        print(f"\nResponse time: {elapsed:.2f} seconds")

    except Exception as error:
        print("Error:", error)


# ============================================================
# Task 3 - Compare llama3.2:3b vs qwen2.5:3b
# ============================================================

comparison_questions = [
    "Explain artificial intelligence in simple words.",
    "What is Retrieval-Augmented Generation?",
    "What are the advantages of using local language models?"
]


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


for number, question in enumerate(
    comparison_questions,
    start=1
):

    print("\n" + "-" * 70)
    print(f"Question {number}: {question}")
    print("-" * 70)

    for model in [
        "llama3.2:3b",
        "qwen2.5:3b"
    ]:

        print(f"\n[{model}]")

        try:

            answer, elapsed = ask_ollama(
                model,
                question
            )

            print(answer)
            print(
                f"Response time: "
                f"{elapsed:.2f} seconds"
            )

        except Exception as error:

            print("Error:", error)


print("\n" + "=" * 70)
print("W7D4 OLLAMA INFERENCE COMPLETED")
print("=" * 70)