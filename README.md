# Hand Glove Defect Detection

This repository contains a medical glove defect detection system built with FastAPI, YOLOv8 (ultralytics), OpenCV and PyTorch. The app exposes a lightweight web interface located in `web_interface/app.py` and performs inference using the trained YOLO model and preprocessing logic in `run_system.py`.

## Deployment

This project is deployed using Docker. A public image is available on Docker Hub:

https://hub.docker.com/r/nawfal18/glove-defect-detector

## Quick start (PowerShell)

Pull and run the published image from Docker Hub:

```powershell
docker pull nawfal18/glove-defect-detector:latest
docker run --rm -p 8000:8000 -v ${PWD}\web_interface\uploads:/app/web_interface/uploads -v ${PWD}\web_interface\results:/app/web_interface\results nawfal18/glove-defect-detector:latest
```

Then open your browser to:

    http://localhost:8000

## Local development with the repo

If you want to build locally from source (see `Dockerfile` and `docker-compose.yml` added to the repo):

```powershell
# Build image locally
docker build -t glove-defect-detector:local .
# Run
docker run --rm -p 8000:8000 -v ${PWD}\web_interface\uploads:/app/web_interface/uploads -v ${PWD}\web_interface\results:/app/web_interface\results glove-defect-detector:local
# or with compose
docker-compose up --build
```

## Notes and caveats

- The container image published on Docker Hub is built with CPU PyTorch wheels by default. If you require GPU/CUDA support, use an appropriate CUDA-enabled base image and install matching PyTorch wheels (or ask me to add a GPU Dockerfile).
- Model weights need to be available at runtime. The code looks for YOLO weights using `utils.config.Config().YOLO_MODEL_DIR` — ensure your `best.pt` (or other weights) are present in the container filesystem or mounted into the correct path.
- The app serves images from `web_interface/uploads` and `web_interface/results`. Those directories are mounted in the run examples so results persist between runs.

## Health check

The app exposes a simple health endpoint:

    GET /health

## Contact / Contributions

If you'd like me to add GPU support, CI to build and push images, or a small production `docker-compose` with Nginx and TLS, tell me which option and I will add it.
