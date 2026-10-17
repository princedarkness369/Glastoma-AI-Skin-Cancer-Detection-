FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU PyTorch wheels first
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.0.1 torchvision==0.15.2

# Install other Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . /app

# Hugging Face Spaces user (UID 1000)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Launch Uvicorn with FastAPI app
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
