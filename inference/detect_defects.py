import torch
import cv2
import numpy as np
import os
import json
from pathlib import Path
from utils.config import Config

class DefectDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)
        self.model.conf = 0.25  # Confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
    
    def detect(self, image_path, output_path=None, save_result=True):
        """Detect defects in an image"""
        # Run inference
        results = self.model(image_path)
        
        # Parse results
        detections = []
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                detections.append({
                    'class': int(cls),
                    'class_name': self.config.CLASS_NAMES[int(cls)],
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        # Save result image if requested
        if save_result and output_path:
            results.render()  # Draw bounding boxes
            cv2.imwrite(output_path, results.ims[0])
        
        return {
            'image_path': image_path,
            'detections': detections,
            'total_defects': len(detections),
            'defect_types': list(set([d['class_name'] for d in detections]))
        }
    
    def detect_batch(self, input_dir, output_dir):
        """Detect defects in all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"detected_{filename}")
                
                try:
                    result = self.detect(input_path, output_path)
                    results.append(result)
                    print(f"Processed: {filename} - Found {result['total_defects']} defects")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "detection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    config = Config()
    
    # Initialize detector
    model_path = os.path.join(config.YOLO_MODEL_DIR, "glove_defect_v1", "weights", "best.pt")
    detector = DefectDetector(model_path, config)
    
    # Detect defects
    input_dir = config.ENHANCED_DATA_DIR  # Or use original images
    output_dir = os.path.join(config.RESULTS_DIR, "detections")
    
    results = detector.detect_batch(input_dir, output_dir)
    print(f"Processed {len(results)} images")

if __name__ == "__main__":
    main()