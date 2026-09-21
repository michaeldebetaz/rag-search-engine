import argparse
from lib.search_utils import (
    build_command,
    search_command,
    tf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency of a word in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("word", type=str, help="Word to search for")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)

        case "build":
            build_command()

        case "tf":
            tf_command(args.doc_id, args.word)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
