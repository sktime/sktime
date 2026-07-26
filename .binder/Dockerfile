# This Dockerfile is used to build sktime when launching binder.
# Find out more at: https://mybinder.readthedocs.io/en/latest/index.html

FROM  jupyter/scipy-notebook:python-3.11.6
# Set up user to avoid running as root
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Binder will automatically clone the repo, but we need to make sure the
# contents of our repo are in the ${HOME} directory
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

# Switch user and directory
USER ${USER}
WORKDIR ${HOME}

# Install extra requirements and sktime based on master branch
RUN pip install --upgrade pip --no-cache-dir && pip install .[binder]
