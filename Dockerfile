# ============================================================
# Dockerfile for Image Diff AI Backend
# ============================================================
# Uses python:3.11-slim for a small footprint (~150MB base).
# Installs system-level dependencies required by OpenCV
# (libgl1 for image rendering, libglib2.0 for GLib).
# ============================================================

FROM python:3.11-slim

# Install system dependencies required by OpenCV
# libgl1-mesa-glx: OpenGL rendering (required by cv2)
# libglib2.0-0: GLib library (required by cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
# (requirements change less often than source code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Create outputs directory for generated visualizations
RUN mkdir -p /app/outputs

# Expose the FastAPI server port
EXPOSE 8000

# Run the FastAPI application with uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
