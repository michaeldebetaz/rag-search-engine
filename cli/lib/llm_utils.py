import base64
import json
import os

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage

from .search_utils import RRFSearchResult, RerankRRFSearchResult, load_env

load_env()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def ask_llm(prompt: str, model: str = "openrouter/free") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Response content is None.")
    usage = response.usage
    if usage is None:
        raise ValueError("Usage information is not available in the response.")
    prompt_tokens = usage.prompt_tokens
    response_tokens = usage.completion_tokens
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Response tokens: {response_tokens}")
    return content.strip()


def correct_query(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    return ask_llm(prompt)


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    return ask_llm(prompt)


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
- "math movie" -> "mathematics mathematician calculus geometry algebra statistics professor genius prodigy theory equation formula proof numbers logic puzzle code cryptography computer genius intellect"

User query: "{query}"
"""
    return ask_llm(prompt)


def rank_invdividual_rrf_result(
    query: str, result: RRFSearchResult
) -> RerankRRFSearchResult:
    doc = result["document"]
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc["title"]} - {doc["description"]}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    score_str = ask_llm(prompt)
    if not score_str.isdigit():
        raise ValueError(f"Expected a numeric score, got: {score_str}")
    score = float(score_str)
    rerank_result: RerankRRFSearchResult = {
        "document": doc,
        "rerank_score": score,
        "bm25_rank": result["bm25_rank"],
        "semantic_rank": result["semantic_rank"],
        "rrf_score": result["rrf_score"],
    }
    return rerank_result


def rank_batch_rrf_results(query: str, rrf_results: list[RRFSearchResult]) -> list[int]:
    doc_list_str = "\n\n".join(
        [
            f"{result['document']['id']}: {result['document']['title']} - {result['document']['description']}"
            for result in rrf_results
        ]
    )

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    ranked_ids: list[int] = []
    json_str = ask_llm(prompt)
    try:
        json_obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from LLM response: {json_str}")
        raise ValueError(f"Failed to parse JSON from LLM response: {e}")

    if not isinstance(json_obj, list):
        raise ValueError(f"Expected a list of integers, got: {json_obj}")
    for item in json_obj:
        if not isinstance(item, int):
            raise ValueError(f"Expected an integer in the list, got: {item}")
        ranked_ids.append(item)

    return ranked_ids


def evaluate_rrf_results(query: str, formatted_rrf_results: str) -> list[int]:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_rrf_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    json_str = ask_llm(prompt)
    try:
        json_obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from LLM response: {json_str}")
        raise ValueError(f"Failed to parse JSON from LLM response: {e}")

    scores: list[int] = []
    if not isinstance(json_obj, list):
        raise ValueError(f"Expected a list, got: {json_obj}")
    for item in json_obj:
        if not isinstance(item, int):
            raise ValueError(f"Expected an integer in the list, got: {item}")
        scores.append(item)
    return scores


def rag_query(query: str, docs: str) -> str:
    prompt = prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
    return ask_llm(prompt)


def summarize_results(query: str, results: str) -> str:
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    return ask_llm(prompt)


def answer_with_citations(query: str, docs: str) -> str:
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    return ask_llm(prompt)


def answer_question(query: str, docs: str) -> str:
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {query}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    return ask_llm(prompt)


def rewrite_query_with_image(
    image: bytes, mime: str, query: str
) -> tuple[str, CompletionUsage]:
    system_prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]
    response = client.chat.completions.create(
        model="openrouter/free", messages=messages
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Response content is None.")
    usage = response.usage
    if usage is None:
        raise ValueError("Usage information is not available in the response.")
    rewritten_query = content.strip()
    return rewritten_query, usage
