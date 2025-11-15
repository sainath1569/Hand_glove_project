from ultralytics import YOLO
import os
import yaml
from utils.config import Config

class YOLOTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def prepare_dataset_yaml(self, dataset_path):
        """Prepare dataset YAML for YOLO"""
        yaml_path = os.path.join(dataset_path, 'glove_dataset.yaml')
        
        dataset_config = {
            'path': os.path.abspath(dataset_path),
            'train': 'train/images',
            'val': 'valid/images', 
            'test': 'test/images' if os.path.exists(os.path.join(dataset_path, 'test/images')) else 'valid/images',
            'nc': self.config.NUM_CLASSES,
            'names': self.config.CLASS_NAMES
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset YAML created: {yaml_path}")
        return yaml_path
    
    def train_yolo(self, dataset_path, epochs=100, batch_size=16):
        """Train YOLO model"""
        yaml_path = self.prepare_dataset_yaml(dataset_path)
        
        # Load YOLO model
        self.model = YOLO('yolov8s.pt')
        
        print(f"Starting YOLO training with dataset: {yaml_path}")
        
        # Train the model
        results = self.model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=15,
            save=True,
            project=self.config.YOLO_MODEL_DIR,
            name='glove_defect_detection',
            exist_ok=True,
            verbose=True
        )
        
        print("YOLO training completed successfully!")
        return results
    
    def train_on_original_dataset(self, dataset_path=None):
        """Train YOLO on original dataset"""
        if dataset_path is None:
            dataset_path = self.config.HAND_GLOVES_DIR
        print("Training YOLO on original dataset...")
        self.train_yolo(dataset_path, epochs=100, batch_size=16)
    
    def train_on_enhanced_dataset(self, enhanced_path=None):
        """Train YOLO on SRGAN enhanced dataset"""
        if enhanced_path is None:
            enhanced_path = os.path.join(self.config.ENHANCED_DATA_DIR, 'yolo_dataset')
        
        print("Training YOLO on enhanced dataset...")
        self.train_yolo(enhanced_path, epochs=100, batch_size=16)
    
    def evaluate_model(self, weights_path=None):
        """Evaluate trained YOLO model"""
        if weights_path is None:
            weights_path = os.path.join(
                self.config.YOLO_MODEL_DIR, 
                'glove_defect_detection/weights/best.pt'
            )
        
        if not os.path.exists(weights_path):
            print(f"Model weights not found at {weights_path}")
            return
        
        model = YOLO(weights_path)
        results = model.val()
        print(f"Evaluation results: {results}")
        return results

def main():
    config = Config()
    trainer = YOLOTrainer(config)
    
    print("YOLO Training Options:")
    print("1. Train on original dataset")
    print("2. Train on enhanced dataset") 
    print("3. Evaluate model")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        trainer.train_on_original_dataset()
    
    elif choice == '2':
        enhanced_path = os.path.join(config.ENHANCED_DATA_DIR, 'yolo_dataset')
        trainer.train_on_enhanced_dataset(enhanced_path)
    
    elif choice == '3':
        trainer.evaluate_model()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()