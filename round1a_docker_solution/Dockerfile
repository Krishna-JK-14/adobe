# Dockerfile (at the top level: round1a_docker_solution/Dockerfile)
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL PART ---
# Copy all your Python application files from the local 'app/' directory
# into the '/app' directory inside the container.
COPY app/ /app/ 

# Copy the run.sh script to the /app directory
COPY run.sh /app/run.sh

# Give execution permissions to the script
RUN chmod +x /app/run.sh

# Define the command that will be executed when the container starts
ENTRYPOINT ["bash", "run.sh"]