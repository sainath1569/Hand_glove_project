import torch
import cv2
import numpy as np
import os
from models.srgan.model import Generator
from utils.config import Config

class ImageEnhancer:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load generator model
        self.generator = Generator(scale_factor=config.SCALE_FACTOR)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.to(self.device)
        self.generator.eval()
    
    def enhance_image(self, image_path, output_path=None):
        """Enhance a single image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original_size = image.shape[:2]
        
        # Preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Enhance
        with torch.no_grad():
            enhanced_tensor = self.generator(image_tensor)
        
        # Convert back to numpy
        enhanced_image = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
        enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        
        # Resize back to original size if needed
        if enhanced_image_bgr.shape[:2] != original_size:
            enhanced_image_bgr = cv2.resize(enhanced_image_bgr, (original_size[1], original_size[0]))
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, enhanced_image_bgr)
        
        return enhanced_image_bgr
    
    def enhance_batch(self, input_dir, output_dir):
        """Enhance all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        enhanced_count = 0
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                try:
                    self.enhance_image(input_path, output_path)
                    enhanced_count += 1
                    print(f"Enhanced: {filename}")
                except Exception as e:
                    print(f"Error enhancing {filename}: {e}")
        
        print(f"Enhanced {enhanced_count} images")
        return enhanced_count

def main():
    config = Config()
    
    # Initialize enhancer
    model_path = os.path.join(config.SRGAN_MODEL_DIR, "srgan_epoch_100.pth")  # Update with your model
    enhancer = ImageEnhancer(model_path, config)
    
    # Enhance images from raw dataset
    input_dir = os.path.join(config.HAND_GLOVES_DIR, "train", "images")
    output_dir = config.ENHANCED_DATA_DIR
    
    enhancer.enhance_batch(input_dir, output_dir)

if __name__ == "__main__":
    main()