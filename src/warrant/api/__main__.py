"""Serve the endpoint.

    python -m warrant.api

Runs the application under uvicorn on the configured host and port. The import string is passed
rather than the app object so that reload -- were it ever enabled -- has a module to re-import; the
default here does not reload, because a server a reviewer runs to check the walking skeleton has no
source to watch change under it.

The host and port are read from the environment with defaults that need no setup, the way every
other runtime value in this project is, so that `make serve` and the container start the same
process with the same knobs.

Uvicorn's own logging is left in place: its default configuration touches only the `uvicorn*`
loggers, so the server's startup banner and access lines still print, while this project's own
`warrant` logger keeps its JSON handler and does not propagate into it. The two coexist without one
reformatting the other.
"""

from __future__ import annotations

import uvicorn

from warrant.settings import get_settings


def main() -> None:
    settings = get_settings()

    uvicorn.run(
        "warrant.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
