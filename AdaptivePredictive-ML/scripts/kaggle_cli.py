"""Invoke the Kaggle CLI with the existing OAuth token without copying secrets.

The wrapper reads the user-owned credential file at runtime, places the access token
only in the child process environment, and never prints or persists it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    credentials_path = Path.home() / ".kaggle" / "credentials.json"
    credentials = json.loads(credentials_path.read_text())
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise SystemExit("Kaggle OAuth access token is missing")
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    executable = Path(sys.executable).with_name("kaggle")
    os.execve(executable, [str(executable), *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()

