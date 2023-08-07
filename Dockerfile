# Use an official Python runtime as a parent image
FROM python:3.9.12-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install sktime, Jupyter, and other dependencies
RUN pip install --no-cache-dir numpy pandas scikit-learn
RUN pip install --no-cache-dir sktime[all_extras]
RUN pip install --no-cache-dir jupyter

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter notebook when the container launches
# CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
