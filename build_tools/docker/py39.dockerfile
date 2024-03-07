FROM python:3.9.18-slim-bookworm

WORKDIR /usr/src/sktime

COPY . .

RUN apt update && apt install -y gcc build-essential git
RUN python -m pip install -U pip
RUN python -m pip install .[all_extras,dev,binder]
