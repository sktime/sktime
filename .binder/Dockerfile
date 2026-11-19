# Use a specific version of the base image for reproducibility
FROM jupyter/scipy-notebook:python-3.11.6

# Set up user to avoid running as root
ARG NB_USER
ARG NB_UID
ENV USER=${NB_USER} HOME=/home/${NB_USER}

# Copy the repository contents into the home directory and adjust ownership in a single layer
COPY --chown=${NB_UID}:${NB_UID} . ${HOME}

# Switch to the non-root user
USER ${USER}
WORKDIR ${HOME}

# Upgrade pip and install dependencies in one RUN command to reduce layers
RUN pip install --upgrade pip --no-cache-dir && \
    pip install .[binder] --no-cache-dir

# Clean up unnecessary files to reduce image size (optional)
RUN find ${HOME} -type d -name __pycache__ -exec rm -rf {} +  # Remove Python cache directories

