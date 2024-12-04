# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the model
COPY . .

# Expose the port for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "controller:app", "--host", "0.0.0.0", "--port", "8000"]