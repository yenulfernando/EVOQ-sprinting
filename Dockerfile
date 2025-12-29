FROM python:3.10-slim

# Prevent Qt / GUI crashes
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

# Set working dir
WORKDIR /app

# Cloud Run output bucket for pose service
ENV OUTPUT_BUCKET=sprint-outputs-q6
ENV PORT=8080

# Install system dependencies needed by OpenCV & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics

# Copy project files
COPY . .

EXPOSE 8080

# Run API
CMD ["python", "app.py"]
