# This Dockerfile is used to build sktime when launching binder.
# Find out more at: https://mybinder.readthedocs.io/en/latest/index.html

# Load pre-existing docker image
FROM jupyter/scipy-notebook:cf6258237ff9

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
RUN conda install -c conda-forge "matplotlib>=3.3.2" "seaborn>=0.11.0" "tsfresh>=0.17.0" "hcrystalball>=0.1.9" "stumpy>=1.5.1" "fbprophet>=0.7.1" "numba>=0.50,<0.53" "numpy>=1.19" "pandas>=1.1.0,<1.2" "statsmodels>=0.12.1" "scikit-learn>=0.23.0" "scikit-posthocs>=0.6.5" pystan
RUN pip install --no-cache-dir -r .binder/requirements.txt
RUN pip install --no-cache-dir .
