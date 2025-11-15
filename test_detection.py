#!/usr/bin/env python3
"""
Standalone test script to debug YOLOv8 detection issues
Run this to see what's happening with model inference
"""

import os
import sys
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import Config

def test_model_loading():
    """Test if model loads correctly"""
    print("=" * 70)
    print("STEP 1: Testing Model Loading")
    print("=" * 70)
    
    config = Config()
    
    possible_paths = [
        os.path.join(config.YOLO_MODEL_DIR, 'glove_defect_detection', 'weights', 'best.pt'),
        os.path.join(config.YOLO_MODEL_DIR, 'glove_defect_rtx', 'weights', 'best.pt'),
        os.path.join(config.YOLO_MODEL_DIR, 'best.pt'),
    ]
    
    print(f"\nLooking for model in YOLO_MODEL_DIR: {config.YOLO_MODEL_DIR}\n")
    
    model_path = None
    for path in possible_paths:
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"{status}: {path}")
        if exists:
            model_path = path
    
    if not model_path:
        print("\n❌ No model found! Cannot proceed with testing.")
        return None
    
    print(f"\n✅ Loading model from: {model_path}")
    
    try:
        model = YOLO(model_path)
        model.conf = 0.25
        print(f"✅ Model loaded successfully!")
        print(f"   Type: {type(model)}")
        print(f"   Confidence threshold set to: {model.conf}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def test_inference(model, image_path, output_dir="test_results"):
    """Test inference on a single image"""
    print("\n" + "=" * 70)
    print("STEP 2: Testing Inference on Image")
    print("=" * 70)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Size: {os.path.getsize(image_path) / 1024:.1f} KB")
    
    # Check image can be read
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image with OpenCV")
        return False
    
    h, w = img.shape[:2]
    print(f"Dimensions: {w}x{h}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Default confidence (usually 0.5)
    print(f"\n🔍 Test 1: Default confidence threshold")
    results_default = model.predict(source=image_path, verbose=False)
    
    print(f"   Boxes found: {len(results_default[0].boxes)}")
    
    # Test 2: Low confidence (0.25)
    print(f"\n🔍 Test 2: Lower confidence threshold (0.25)")
    results_low = model.predict(source=image_path, conf=0.25, verbose=False)
    
    print(f"   Boxes found: {len(results_low[0].boxes)}")
    
    # Test 3: Very low confidence (0.1) - catch everything
    print(f"\n🔍 Test 3: Very low confidence threshold (0.1)")
    results_verylow = model.predict(source=image_path, conf=0.1, verbose=False)
    
    print(f"   Boxes found: {len(results_verylow[0].boxes)}")
    
    # Show detections from best result
    best_results = results_low if len(results_low[0].boxes) > 0 else results_verylow
    
    print(f"\n📊 Detections from conf=0.25:")
    if len(best_results[0].boxes) > 0:
        for idx, box in enumerate(best_results[0].boxes):
            cls = int(box.cls[0]) if len(box.cls) > 0 else 0
            conf = float(box.conf[0]) if len(box.conf) > 0 else 0.0
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"   {idx+1}. Class={cls}, Conf={conf:.4f}, BBox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        print("   ⚠️  No detections found at conf=0.25")
        print("   This indicates a model training issue or insufficient data.")
    
    # Save annotated images
    for conf_val, res in [(0.25, results_low), (0.1, results_verylow)]:
        for r in res:
            im_array = r.plot()
            out_path = os.path.join(output_dir, f"test_detected_conf{int(conf_val*100)}.jpg")
            cv2.imwrite(out_path, im_array)
            print(f"\n   📸 Saved: {out_path}")
            break
    
    return len(best_results[0].boxes) > 0


def test_config():
    """Test configuration"""
    print("\n" + "=" * 70)
    print("STEP 3: Checking Configuration")
    print("=" * 70)
    
    config = Config()
    print(f"\nClass Names: {config.CLASS_NAMES}")
    print(f"Num Classes: {config.NUM_CLASSES}")
    print(f"YOLO Model Dir: {config.YOLO_MODEL_DIR}")
    print(f"HAND_GLOVES_DIR: {config.HAND_GLOVES_DIR}")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "YOLOv8 GLOVE DEFECT DETECTION TEST SUITE" + " " * 12 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Step 1: Load model
    model = test_model_loading()
    if not model:
        print("\n❌ Cannot continue without model")
        return
    
    # Step 2: Test config
    test_config()
    
    # Step 3: Find a test image
    print("\n" + "=" * 70)
    print("STEP 4: Finding Test Image")
    print("=" * 70)
    
    config = Config()
    test_image = None
    
    # Look for test image in common locations
    search_paths = [
        os.path.join(config.HAND_GLOVES_DIR, "test", "images"),
        os.path.join(config.HAND_GLOVES_DIR, "train", "images"),
        os.path.join(config.HAND_GLOVES_DIR, "valid", "images"),
        "test_image.jpg",
        "data/raw/hand_gloves/test/images",
    ]
    
    for search_path in search_paths:
        if os.path.isdir(search_path):
            images = [f for f in os.listdir(search_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(search_path, images[0])
                print(f"\n✅ Found test image: {test_image}")
                break
        elif os.path.isfile(search_path):
            test_image = search_path
            print(f"\n✅ Found test image: {test_image}")
            break
    
    if not test_image:
        print("\n⚠️  No test image found. Provide an image path as argument:")
        print(f"   python test_detection.py path/to/glove/image.jpg")
        
        # If command line arg provided, use that
        if len(sys.argv) > 1:
            test_image = sys.argv[1]
            print(f"\nUsing image from command line: {test_image}")
        else:
            print("\n❌ Cannot proceed without test image")
            return
    
    # Step 4: Run inference
    success = test_inference(model, test_image)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if success:
        print("\n✅ Model is detecting objects correctly!")
        print("   The model appears to be working properly.")
    else:
        print("\n❌ Model found NO detections")
        print("\nPossible causes:")
        print("   1. Model is not properly trained (weights may be random)")
        print("   2. Image does not contain actual defects")
        print("   3. Image format or quality issue")
        print("   4. Defects are too small or unclear")
        print("\nSolutions:")
        print("   1. Retrain the model with better labeled data")
        print("   2. Use labeled validation images with known defects")
        print("   3. Check image pre-processing steps")
        print("   4. Adjust confidence threshold lower")
    
    print("\n📁 Test results saved to: test_results/\n")


if __name__ == "__main__":
    main()
