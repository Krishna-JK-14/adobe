# Use the specified base image for Python 3.10 on a Linux AMD64 platform
FROM --platform=linux/amd64 python:3.10

# Set the working directory inside the container to /app
# All subsequent commands will be executed relative to this directory.
WORKDIR /app

# Copy the main Python script into the container's /app directory
COPY extractor.py .

# Copy the requirements.txt file into the container's /app directory
COPY requirements.txt .

# Copy the pre-downloaded NLP model and its associated files into the container.
# This ensures the model is available within the image at /app/models/.
COPY models/ ./models/

# Install the Python dependencies listed in requirements.txt.
# --no-cache-dir option is used to prevent pip from storing cached data,
# which helps in keeping the Docker image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Define the command that will be executed when the container starts.
# The extractor.py script is designed to expect input.json in /app/input
# and write output to /app/output due to volume mounts provided at runtime.
# It takes no command-line arguments here, as input.json path is hardcoded inside.
CMD ["python", "extractor.py"]