# Use an official Python runtime as a parent image
FROM python:3.11-slim 
# Or python:3.10-slim if you used that

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
# Ensure .dockerignore prevents copying venv, .env etc.
COPY . .

# Expose the port the app runs on (Hugging Face Spaces often uses 7860)
EXPOSE 7860

# Define the command to run the application
# Use the port expected by Hugging Face Spaces (usually 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]