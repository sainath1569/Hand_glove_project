from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from run_system import GloveDefectDetectionSystem
from utils.config import Config

app = FastAPI(title="Medical Glove Defect Detection System")

# Get absolute paths
web_interface_dir = Path(__file__).parent
static_dir = web_interface_dir / "static"
templates_dir = web_interface_dir / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize the detection system
system = GloveDefectDetectionSystem()

# Create uploads and results directories with absolute paths
UPLOAD_DIR = str(web_interface_dir / "uploads")
RESULTS_DIR = str(web_interface_dir / "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the beautiful Tailwind-styled web interface"""
    template_path = templates_dir / "index.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get('/health')
async def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Process the image
    try:
        result = system.process_single_image(file_path, RESULTS_DIR)
        
        if result:
            # Convert file paths to URLs for frontend
            enhanced_name = result.get('enhanced_image', '')
            response_results = {
                'original_image': f"/images/{os.path.basename(file_path)}",
                'detected_image': f"/images/{os.path.basename(result.get('detected_image', ''))}",
                'enhanced_image': f"/images/{os.path.basename(enhanced_name)}" if enhanced_name else '',
                'detections': result.get('detections', []),
                'total_defects': result.get('total_defects', 0),
                'detection_performed': result.get('detection_performed', False)
            }
            
            return {
                "status": "success",
                "message": "Image processed successfully",
                "results": response_results
            }
        else:
            raise HTTPException(status_code=500, detail="Error processing image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Serve images from both uploads and results directories"""
    # Try results directory first (detected images)
    result_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(result_path):
        return FileResponse(result_path, media_type="image/jpeg")
    
    # Try uploads directory
    upload_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(upload_path):
        return FileResponse(upload_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/results/{filename}")
async def get_result(filename: str):
    result_path = os.path.join(RESULTS_DIR, f"{filename}_result.json")
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Result not found")

@app.get("/images/{image_type}/{filename}")
async def get_image_typed(image_type: str, filename: str):
    image_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)