FROM python:3.10.10-bullseye

WORKDIR /usr/src/sktime

COPY . .

RUN python -m pip install -U pip
RUN python -m pip install .[all_extras,dev,binder]
