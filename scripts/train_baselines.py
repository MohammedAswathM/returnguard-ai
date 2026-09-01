#!/usr/bin/env python3
from returnguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-baselines", *(__import__("sys").argv[1:])]))

