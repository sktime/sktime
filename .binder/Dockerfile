# Use pre-existing docker image
FROM jupyter/scipy-notebook:cf6258237ff9
#FROM python:3.7-slim

#ENV DEBIAN_FRONTEND=noninteractive

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

# Install extra requirements and latest version
RUN pip install --upgrade pip --no-cache-dir -r .binder/requirements.txt
RUN pip install --no-cache-dir .
