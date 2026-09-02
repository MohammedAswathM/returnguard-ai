#!/usr/bin/env python3
import os

import uvicorn

from returnguard.backend.api import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
