# Use pre-existing docker image
#FROM jupyter/scipy-notebook:cf6258237ff9
FROM python:3.7-slim

ENV DEBIAN_FRONTEND=noninteractive

# Set up user to avoid running as root
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

#RUN groupadd --gid ${NB_UID} ${NB_USER} && useradd --comment "Default user" --create-home --gid ${NB_UID} --no-log-init --shell /bin/bash --uid ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${USERR}
WORKDIR ${HOME}

## Clone repo
#ARG REPO_DIR=${HOME}
#ENV REPO_DIR ${REPO_DIR}
#WORKDIR ${REPO_DIR}
#RUN git clone https://github.com/alan-turing-institute/sktime.git
#WORKDIR ${REPO_DIR}/sktime

# Check out master branch
#RUN git checkout master

# Install latest version
RUN pip install --no-cache-dir -r build_tools/requirements.txt
RUN pip install --no-cache-dir .

# Install exta requirements for notebooks
RUN pip install --no-cache-dir -r .binder/requirements.txt
