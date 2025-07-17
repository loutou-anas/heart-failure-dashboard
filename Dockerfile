# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Streamlit app and models
COPY dashboard/app.py ./dashboard/app.py
COPY models/ ./models/
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
