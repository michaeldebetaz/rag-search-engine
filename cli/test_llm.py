import os

from openai import OpenAI
from lib.search_utils import load_env

load_env()


def main():
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
            }
        ],
    )
    content = response.choices[0].message.content
    usage = response.usage
    if usage is None:
        raise ValueError("Usage information is not available in the response.")
    prompt_tokens = usage.prompt_tokens
    response_tokens = usage.completion_tokens

    print(f"Content: {content}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Response tokens: {response_tokens}")


if __name__ == "__main__":
    main()
