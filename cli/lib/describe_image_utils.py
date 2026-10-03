import mimetypes

from .llm_utils import rewrite_query_with_image


def describe_image_command(image_path: str, query: str) -> None:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"
    image = open(image_path, "rb").read()
    rewritten_query, usage = rewrite_query_with_image(image, mime, query)
    print(f"Rewritten query: {rewritten_query}")
    print(f"Total tokens: {usage.total_tokens}")
