FROM python:3.10.13-slim-bookworm

WORKDIR /home/skd/Workspace/sktime

COPY . .
RUN apt update && apt install -y gcc
RUN python -m pip install -U pip
RUN python -m pip install ."[all_extras,dev,binder]"
