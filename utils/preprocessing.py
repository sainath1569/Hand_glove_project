import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

class GloveDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.ENHANCED_DATA_DIR, exist_ok=True)
    
    def load_dataset_info(self, yaml_path):
        """Load dataset information from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    
    def create_low_res_images(self, image_dir, save_dir, scale_factor=4):
        """Create low-resolution versions of images for SRGAN training"""
        os.makedirs(save_dir, exist_ok=True)
        
        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                img_path = os.path.join(image_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Create low-resolution version
                    h, w = img.shape[:2]
                    lr_h, lr_w = h // scale_factor, w // scale_factor
                    
                    # Downscale
                    lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
                    
                    # Upscale to original size (this will be our input for SRGAN)
                    lr_upscaled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
                    
                    # Save low-res and high-res pairs
                    base_name = os.path.splitext(img_name)[0]
                    cv2.imwrite(os.path.join(save_dir, f"{base_name}_lr.jpg"), lr_upscaled)
                    cv2.imwrite(os.path.join(save_dir, f"{base_name}_hr.jpg"), img)
        
        print(f"Created low-res/high-res pairs in {save_dir}")
    
    def prepare_srgan_dataset(self, train_dir, valid_dir):
        """Prepare dataset for SRGAN training"""
        print("Preparing SRGAN dataset...")
        
        # Create training pairs
        train_srgan_dir = os.path.join(self.config.PROCESSED_DATA_DIR, "srgan_train")
        self.create_low_res_images(
            os.path.join(train_dir, "images"),
            train_srgan_dir,
            self.config.SCALE_FACTOR
        )
        
        # Create validation pairs
        valid_srgan_dir = os.path.join(self.config.PROCESSED_DATA_DIR, "srgan_valid")
        self.create_low_res_images(
            os.path.join(valid_dir, "images"),
            valid_srgan_dir,
            self.config.SCALE_FACTOR
        )
        
        return train_srgan_dir, valid_srgan_dir

if __name__ == "__main__":
    from config import Config
    preprocessor = GloveDataPreprocessor(Config())