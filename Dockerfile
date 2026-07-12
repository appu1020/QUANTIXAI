FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for scikit-learn, building wheels, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . /app

# Ensure start.sh is executable
RUN chmod +x /app/start.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV DEBUG=False
ENV ALLOWED_HOSTS="*,localhost,127.0.0.1"

# Expose port 8000
EXPOSE 8000

# Run the startup script
CMD ["bash" , "/app/start.sh"]
