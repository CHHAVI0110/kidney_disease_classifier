# ---------- Base image ----------
FROM python:3.10-slim-bullseye

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# ---------- Install minimal system dependencies ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Copy only requirements first ----------
COPY requirements.txt /app/requirements.txt

# ---------- Install Python dependencies ----------
# Install CPU-only TensorFlow first
RUN pip install --no-cache-dir --default-timeout=200 tensorflow-cpu==2.16.1



# Install DVC separately (to avoid dulwich hash mismatch)
RUN pip install --no-cache-dir --default-timeout=200 dvc

# Install remaining dependencies (NO MIRRORS)
RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt

# ---------- Copy application code ----------
COPY . /app

# Install your src package
RUN pip install -e .

# ---------- Expose port ----------
EXPOSE 5000

# ---------- Run FastAPI ----------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
