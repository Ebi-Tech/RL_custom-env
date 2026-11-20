"""Entry point: uv run main.py starts the 3D market street (play.py) with its defaults.

Any arguments are passed through, for example: uv run main.py --market harsh --seed 1003
"""
import sys

from play import main as play_main


def main():
    """Start play.py with the command line arguments given to main.py."""
    play_main(sys.argv[1:])


if __name__ == "__main__":
    main()
