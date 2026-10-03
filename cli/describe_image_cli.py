import argparse

from lib.describe_image_utils import describe_image_command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe an image to assign it to a movie"
    )
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--query", type=str, help="Query to rewrite based on the image")

    args = parser.parse_args()

    describe_image_command(args.image, args.query)


if __name__ == "__main__":
    main()
