import os
import sys
from utils.config import Config
from ultralytics import YOLO
import cv2
import json
import numpy as np

class GloveDefectDetectionSystem:
    def __init__(self):
        self.config = Config()
        self.srgan_model = None
        self.yolo_model = self.load_yolo_model()
        print("Medical Glove Defect Detection System initialized!")
        print("System ready for inference.")
    
    def load_yolo_model(self):
        """Load the trained YOLOv8 model"""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(self.config.YOLO_MODEL_DIR, 'glove_defect_detection', 'weights', 'best.pt'),
            os.path.join(self.config.YOLO_MODEL_DIR, 'glove_defect_rtx', 'weights', 'best.pt'),
            os.path.join(self.config.YOLO_MODEL_DIR, 'best.pt'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            try:
                model = YOLO(model_path)
                # Set confidence threshold low to catch all potential detections
                model.conf = 0.25  # Default YOLOv8 threshold
                print(f"✅ YOLOv8 model loaded successfully from: {model_path}")
                print(f"   Confidence threshold: {model.conf}")
                return model
            except Exception as e:
                print(f"❌ Error loading YOLOv8 model: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("❌ YOLOv8 model not found at expected paths:")
            for path in possible_paths:
                print(f"   - {path}")
            return None
    
    def process_single_image(self, image_path, output_dir="results"):
        """Process a single image through the detection pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(output_dir, f"{base_name}_result.json")
        
        # Copy original image to results directory for serving
        original_output_path = os.path.join(output_dir, f"original_{os.path.basename(image_path)}")
        import shutil
        try:
            shutil.copy(image_path, original_output_path)
        except Exception as e:
            print(f"⚠️  Could not copy original image: {e}")
        
        results = {
            'original_image': f"original_{os.path.basename(image_path)}",
            'enhancement_applied': False,
            'detection_performed': False,
            'detections': [],
            'total_defects': 0
        }
        
        try:
            # Step 1: Detect defects using YOLOv8
            if self.yolo_model:
                detection_result = self.detect_defects_yolov8(image_path, output_dir)
                results.update(detection_result)
                results['detection_performed'] = True
                results['detected_image'] = detection_result.get('detected_image', '')
            
            # Save results
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Processing completed. Results saved to {result_path}")
            return results
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None
    
    def detect_defects_yolov8(self, image_path, output_dir):
        if not self.yolo_model:
            return {'detections': [], 'total_defects': 0}

        print(f"🔍 Preprocessing + YOLOv8 detection on: {os.path.basename(image_path)}")

        # -----------------------------
        # STEP 1: READ IMAGE
        # -----------------------------
        img = cv2.imread(image_path)

        if img is None:
            print("❌ Error: Could not read image.")
            return {'detections': [], 'total_defects': 0}

        # -----------------------------
        # STEP 2: ENHANCE DEFECT VISIBILITY
        # -----------------------------
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks for different defect colors
        # Red stains (multiple ranges for red in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Brown stains (common in defects)
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Dark stains (any dark areas that aren't part of normal glove)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Yellow stains (common discoloration)
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine all defect masks
        defect_mask = cv2.bitwise_or(red_mask, brown_mask)
        defect_mask = cv2.bitwise_or(defect_mask, dark_mask)
        defect_mask = cv2.bitwise_or(defect_mask, yellow_mask)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        
        # -----------------------------
        # STEP 3: CREATE ENHANCED IMAGE WITH BLACK DEFECTS
        # -----------------------------
        # Convert original image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to grayscale for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Convert back to 3-channel
        enhanced_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # Set defect areas to black in the enhanced image
        enhanced_color[defect_mask > 0] = [0, 0, 0]  # Set defects to black
        
        # Save the enhanced image with black defects
        preprocessed_path = os.path.join(output_dir, f"preprocessed_{os.path.basename(image_path)}")
        cv2.imwrite(preprocessed_path, enhanced_color)
        print(f"   🖼️ Preprocessed image with black defects saved: {preprocessed_path}")
        
        # Save the defect mask for debugging
        mask_path = os.path.join(output_dir, f"defect_mask_{os.path.basename(image_path)}")
        cv2.imwrite(mask_path, defect_mask)
        print(f"   🎭 Defect mask saved: {mask_path}")
        
        # -----------------------------
        # STEP 4: YOLO INFERENCE
        # -----------------------------
        results = self.yolo_model.predict(source=preprocessed_path, conf=0.25, iou=0.45, verbose=True)

        detections = []
        detection_count = 0

        for r in results:
            print(f"   Total boxes detected: {len(r.boxes)}")

            for idx, box in enumerate(r.boxes):
                cls = int(box.cls[0]) if len(box.cls) > 0 else 0
                conf = float(box.conf[0]) if len(box.conf) > 0 else 0.0

                if conf >= 0.25:
                    detection_count += 1
                    print(f"   Detection {detection_count}: class={cls} ({self.config.CLASS_NAMES[cls]}), conf={conf:.2f}")

                    detections.append({
                        'class': cls,
                        'class_name': self.config.CLASS_NAMES[cls],
                        'confidence': conf,
                        'bbox': box.xyxy[0].tolist()
                    })

        if detection_count == 0:
            print("   ⚠️ No detections found after preprocessing.")
        else:
            print(f"   ✅ Found {detection_count} defect(s)")

        # -----------------------------
        # STEP 5: SAVE ANNOTATED IMAGE
        # -----------------------------
        detected_image_filename = ""
        for r in results:
            im_array = r.plot()
            detected_image_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(detected_image_path, im_array)
            detected_image_filename = f"detected_{os.path.basename(image_path)}"
            print(f"   📸 Annotated image saved: {detected_image_path}")
            break

        return {
            'detections': detections,
            'total_defects': len(detections),
            'detected_image': detected_image_filename,
            'preprocessed_image': f"preprocessed_{os.path.basename(image_path)}",
            'defect_mask': f"defect_mask_{os.path.basename(image_path)}"
        }

def main():
    system = GloveDefectDetectionSystem()
    print("🚀 Launching web interface...")
    print("🌐 Open http://127.0.0.1:8000 in your browser")
    os.system('cd web_interface && python app.py')

if __name__ == "__main__":
    main()