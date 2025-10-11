FROM python:3.10-slim

WORKDIR /usr/src/sktime

COPY . .

RUN apt-get update && apt-get install --no-install-recommends -y build-essential gcc git && apt-get clean
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install -U pip
RUN uv pip install .[all_extras_pandas2,dev,binder]
