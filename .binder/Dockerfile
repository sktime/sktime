# This Dockerfile is used to build sktime when launching binder.
# Find out more at: https://mybinder.readthedocs.io/en/latest/index.html

# Load jupyter python 3.8 image
# 3.8 is the highest currently supported version we can use
FROM  jupyter/scipy-notebook:python-3.8.8
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
USER ${USERR}
WORKDIR ${HOME}

# Install extra requirements and sktime based on master branch
RUN pip install --upgrade pip --no-cache-dir -r .binder/prelim_requirements.txt
RUN pip install --upgrade pip --no-cache-dir -r .binder/requirements.txt
RUN pip install --no-cache-dir .
