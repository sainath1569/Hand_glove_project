# Deployment quickstart

This repository contains a FastAPI app in `web_interface/app.py` that runs a glove defect detection pipeline (YOLOv8 + preprocessing). Below are quick instructions to build a Docker image and run locally. These are minimal, CPU-friendly steps; adjust for GPU if needed.

1) Build the Docker image (Windows PowerShell):

```powershell
docker build -t glove-defect-detector:latest .
```

2) Run with Docker (exposes port 8000):

```powershell
docker run --rm -p 8000:8000 -v ${PWD}\web_interface\uploads:/app/web_interface/uploads -v ${PWD}\web_interface\results:/app/web_interface\results glove-defect-detector:latest
```

3) Or use docker-compose for local development (recommended):

```powershell
docker-compose up --build
```

4) Open the app in your browser:

    http://localhost:8000

Notes and caveats:
- The Dockerfile installs the CPU builds of PyTorch. If you have an NVIDIA GPU and want CUDA support, replace the PyTorch install line in the Dockerfile with the correct CUDA index URL or use an NVIDIA CUDA base image and the nvidia-container-toolkit.
- Model weights (YOLO) are expected to be under the paths defined by `utils.config.Config().YOLO_MODEL_DIR`. Ensure your trained weights are copied into the container (or mounted) at runtime.
- Container size will be large because of PyTorch and model weights.

Next steps (optional):
- Push the built image to Docker Hub / GitHub Container Registry and deploy to a cloud service (Cloud Run, AWS ECS, Azure App Service for Containers, or a Kubernetes cluster).
- Add a small Nginx service in `docker-compose` if you want to serve static files more efficiently or add TLS.
