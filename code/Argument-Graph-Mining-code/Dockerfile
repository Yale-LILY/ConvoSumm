# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker

FROM python:3.7-slim
ENV POETRY_VERSION=1.0.0

WORKDIR /app

RUN apt update \
    && apt install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

RUN python -m nltk.downloader punkt stopwords \
    && python -m spacy download en_core_web_lg \
    && python -m spacy download de_core_news_md
