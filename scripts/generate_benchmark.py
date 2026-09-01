#!/usr/bin/env python3
from returnguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate-data", *(__import__("sys").argv[1:])]))

