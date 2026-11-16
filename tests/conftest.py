"""Loaded by pytest before any test file. The tests run with the defaults in similarity/config.py,
whatever the local .env says, so editing .env never makes a test fail."""

import os

os.environ["SIMILARITY_ENV_FILE"] = ""
