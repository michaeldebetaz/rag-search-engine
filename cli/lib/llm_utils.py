import json
import os

from openai import OpenAI

from .search_utils import RRFSearchResult, RerankRRFSearchResult, load_env

load_env()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


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
