# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Ensure model.pkl and label_encoder.pkl is in the correct location
COPY ml_logic/model.pkl /app/ml_logic/model.pkl
COPY ml_logic/label_encoder.pkl /app/ml_logic/label_encoder.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # Expose the port your app uses (if needed)
# EXPOSE 8000

# Command to run your application
CMD uvicorn ml_logic.fast:app --host 0.0.0.0 --port $PORT
