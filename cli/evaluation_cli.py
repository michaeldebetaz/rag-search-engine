import argparse

from lib.evaluation import evaluation_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Number of results to consider for RRF (k for RRF)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    evaluation_command(args.k, limit)


if __name__ == "__main__":
    main()
