FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip and wheel first
RUN python -m pip install --upgrade pip wheel setuptools

# Install PyTorch CPU wheels explicitly (faster, avoids CUDA mismatches). \
# If you want GPU support, replace this with the appropriate CUDA-index URL or use an NVIDIA base image.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose the port the FastAPI app uses
EXPOSE 8000

# Run the app with Uvicorn (production: consider using multiple workers / a process manager)
CMD ["uvicorn", "web_interface.app:app", "--host", "0.0.0.0", "--port", "8000"]
