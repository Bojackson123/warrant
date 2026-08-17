"""The HTTP surface: a question in, an answer and its citations out.

One endpoint, `POST /answer`, assembled in `app`. The request path is the recorder's path run
forwards -- retrieve, render, generate-or-decline -- and lives in `pipeline`; the two-half citation
check that gives the answer its warrant lives in `citations`; the shapes that cross the boundary are
in `schemas`. `python -m warrant.api` serves it.
"""
