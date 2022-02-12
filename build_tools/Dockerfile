FROM python:3.9-buster

WORKDIR /usr/src/sktime

COPY . .

RUN python -m pip install -U pip
RUN python -m pip install .[all_extras,dev]
