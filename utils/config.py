import os

# Get the project root directory (parent of this utils directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Dataset paths (absolute)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    ENHANCED_DATA_DIR = os.path.join(DATA_DIR, "enhanced")
    HAND_GLOVES_DIR = os.path.join(RAW_DATA_DIR, "hand_gloves")
    
    # Model paths (absolute)
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    SRGAN_MODEL_DIR = os.path.join(MODELS_DIR, "srgan")
    YOLO_MODEL_DIR = os.path.join(MODELS_DIR, "yolo")
    
    # Results and training paths
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    IMAGE_SIZE = 256
    
    # SRGAN specific
    SCALE_FACTOR = 4  # 4x super resolution
    
    # YOLO classes
    CLASS_NAMES = ['large', 'medium', 'small']
    NUM_CLASSES = 3
    
config = Config()