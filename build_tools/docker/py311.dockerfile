FROM python:3.11-slim

WORKDIR /usr/src/sktime

COPY . .

RUN apt-get update && apt-get install --no-install-recommends -y build-essential gcc git && apt-get clean
RUN python -m pip install -U pip
RUN python -m pip install .[all_extras_pandas2,dev,binder]
