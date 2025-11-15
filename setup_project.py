import os

def create_all_files():
    print("🚀 Creating complete Medical Glove Defect Detection System...")
    
    # 1. Create SRGAN Model
    print("📁 Creating SRGAN model...")
    srgan_model_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Upsampling blocks
        upsample_blocks = []
        for _ in range(scale_factor // 2):
            upsample_blocks.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ])
        self.upsample = nn.Sequential(*upsample_blocks)
        
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x = self.res_blocks(x1)
        x = self.bn2(self.conv2(x))
        x += x1
        x = self.upsample(x)
        x = torch.tanh(self.conv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)
'''
    with open('models/srgan/model.py', 'w') as f:
        f.write(srgan_model_code)

    # 2. Create SRGAN Training
    print("📁 Creating SRGAN training...")
    srgan_training_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.srgan.model import Generator, Discriminator
from utils.config import Config

class GloveDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()
    
    def _load_image_pairs(self):
        pairs = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_lr.jpg'):
                lr_path = os.path.join(self.data_dir, file)
                hr_path = os.path.join(self.data_dir, file.replace('_lr.jpg', '_hr.jpg'))
                if os.path.exists(hr_path):
                    pairs.append((lr_path, hr_path))
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        lr_path, hr_path = self.image_pairs[idx]
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        else:
            transform = transforms.ToTensor()
            lr_image = transform(lr_image)
            hr_image = transform(hr_image)
        
        return lr_image, hr_image

class SRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.generator = Generator(scale_factor=config.SCALE_FACTOR).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        self.adversarial_criterion = nn.BCELoss()
        self.content_criterion = nn.L1Loss()
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config.LEARNING_RATE)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.LEARNING_RATE)
        
        self.g_losses = []
        self.d_losses = []
        
        os.makedirs(config.SRGAN_MODEL_DIR, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs=100):
        print("Starting SRGAN training...")
        
        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()
            
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                real_outputs = self.discriminator(hr_imgs)
                real_loss = self.adversarial_criterion(real_outputs, torch.ones_like(real_outputs))
                
                fake_imgs = self.generator(lr_imgs)
                fake_outputs = self.discriminator(fake_imgs.detach())
                fake_loss = self.adversarial_criterion(fake_outputs, torch.zeros_like(fake_outputs))
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                fake_outputs = self.discriminator(fake_imgs)
                adversarial_loss = self.adversarial_criterion(fake_outputs, torch.ones_like(fake_outputs))
                content_loss = self.content_criterion(fake_imgs, hr_imgs)
                
                g_loss = adversarial_loss + 0.001 * content_loss
                g_loss.backward()
                self.g_optimizer.step()
                
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}')
            
            avg_g_loss = g_loss_epoch / len(train_loader)
            avg_d_loss = d_loss_epoch / len(train_loader)
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
            
            if (epoch + 1) % 20 == 0:
                self.save_model(epoch + 1)
    
    def save_model(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }
        model_path = os.path.join(self.config.SRGAN_MODEL_DIR, f'srgan_epoch_{epoch}.pth')
        torch.save(checkpoint, model_path)
        print(f"Model saved: {model_path}")

def main():
    config = Config()
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    train_data_dir = "data/processed/srgan_train"
    val_data_dir = "data/processed/srgan_valid"
    
    train_dataset = GloveDataset(train_data_dir, transform=transform)
    val_dataset = GloveDataset(val_data_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    trainer = SRGANTrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
'''
    with open('training/srgan_training.py', 'w') as f:
        f.write(srgan_training_code)

    # 3. Create YOLO Training
    print("📁 Creating YOLO training...")
    yolo_training_code = '''import torch
import yaml
import os
import subprocess
import sys
from utils.config import Config

class YOLOTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_yolo()
    
    def setup_yolo(self):
        if not os.path.exists('yolov5'):
            print("Cloning YOLOv5 repository...")
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], check=True)
        
        print("Installing YOLOv5 requirements...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'yolov5/requirements.txt'], check=True)
        print("YOLOv5 setup completed!")
    
    def create_dataset_yaml(self, dataset_path):
        yolo_data = {
            'path': dataset_path,
            'train': 'train/images',
            'val': 'valid/images',
            'nc': self.config.NUM_CLASSES,
            'names': self.config.CLASS_NAMES
        }
        
        yaml_path = os.path.join(dataset_path, 'glove_dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yolo_data, f)
        return yaml_path
    
    def train(self, dataset_path, epochs=100, batch_size=16):
        yaml_path = self.create_dataset_yaml(dataset_path)
        
        print(f"Starting YOLOv5 training with dataset: {yaml_path}")
        
        cmd = [
            sys.executable, 'yolov5/train.py',
            '--img', '640',
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', yaml_path,
            '--weights', 'yolov5s.pt',
            '--project', self.config.YOLO_MODEL_DIR,
            '--name', 'glove_defect_detection',
            '--exist-ok'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("YOLO training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"YOLO training failed: {e}")

def main():
    config = Config()
    trainer = YOLOTrainer(config)
    dataset_path = "hand_gloves"
    trainer.train(dataset_path)

if __name__ == "__main__":
    main()
'''
    with open('training/yolo_training.py', 'w') as f:
        f.write(yolo_training_code)

    # 4. Create Web Interface
    print("📁 Creating web interface...")
    web_app_code = '''from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import uuid
import shutil

app = FastAPI(title="Medical Glove Defect Detection System")
app.mount("/static", StaticFiles(directory="web_interface/static"), name="static")

UPLOAD_DIR = "web_interface/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Medical Glove Defect Detection</title>
            <style>
                body { font-family: Arial; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                .btn { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Medical Glove Defect Detection System</h1>
                <p>Upload a medical glove image to detect defects</p>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <div class="upload-area">
                        <input type="file" name="file" accept="image/*" required>
                    </div>
                    <button type="submit" class="btn">Detect Defects</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return {
        "status": "success",
        "message": "File uploaded successfully",
        "file_path": file_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    with open('web_interface/app.py', 'w') as f:
        f.write(web_app_code)

    # 5. Create Inference Files
    print("📁 Creating inference files...")
    enhance_code = '''import torch
import cv2
import numpy as np
import os
from models.srgan.model import Generator
from utils.config import Config

class ImageEnhancer:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(scale_factor=config.SCALE_FACTOR)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.to(self.device)
        self.generator.eval()
    
    def enhance_image(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            enhanced_tensor = self.generator(image_tensor)
        
        enhanced_image = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
        enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        
        if output_path:
            cv2.imwrite(output_path, enhanced_image_bgr)
        
        return enhanced_image_bgr

print("Image enhancer created successfully!")
'''
    with open('inference/enhance_images.py', 'w') as f:
        f.write(enhance_code)

    detect_code = '''import torch
import cv2
import numpy as np
import os
from utils.config import Config

class DefectDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)
        self.model.conf = 0.25
    
    def detect(self, image_path, output_path=None):
        results = self.model(image_path)
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
        
        if output_path:
            results.render()
            cv2.imwrite(output_path, results.ims[0])
        
        return {
            'image_path': image_path,
            'detections': detections,
            'total_defects': len(detections)
        }

print("Defect detector created successfully!")
'''
    with open('inference/detect_defects.py', 'w') as f:
        f.write(detect_code)

    # 6. Create Main Runner
    print("📁 Creating main system runner...")
    main_runner_code = '''import os
from utils.config import Config

class GloveDefectDetectionSystem:
    def __init__(self):
        self.config = Config()
        print("Medical Glove Defect Detection System initialized!")
        print("System ready for training and inference.")
    
    def show_menu(self):
        print("\\n🎯 Medical Glove Defect Detection System")
        print("1. Train SRGAN (Image Enhancement)")
        print("2. Train YOLO (Defect Detection)") 
        print("3. Run Web Interface")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        return choice

def main():
    system = GloveDefectDetectionSystem()
    
    while True:
        choice = system.show_menu()
        
        if choice == '1':
            print("Starting SRGAN training...")
            os.system('python training/srgan_training.py')
        
        elif choice == '2':
            print("Starting YOLO training...")
            os.system('python training/yolo_training.py')
        
        elif choice == '3':
            print("Starting web interface...")
            print("Open http://localhost:8000 in your browser")
            os.system('python web_interface/app.py')
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
'''
    with open('run_system.py', 'w') as f:
        f.write(main_runner_code)

    # 7. Create Preprocessing
    print("📁 Creating preprocessing utilities...")
    preprocessing_code = '''import os
import cv2
import numpy as np
from utils.config import Config

class GloveDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
    
    def setup_directories(self):
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.ENHANCED_DATA_DIR, exist_ok=True)
        print("Directories created successfully!")
    
    def create_low_res_images(self, image_dir, save_dir, scale_factor=4):
        os.makedirs(save_dir, exist_ok=True)
        
        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    h, w = img.shape[:2]
                    lr_h, lr_w = h // scale_factor, w // scale_factor
                    
                    lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
                    lr_upscaled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
                    
                    base_name = os.path.splitext(img_name)[0]
                    cv2.imwrite(os.path.join(save_dir, f"{base_name}_lr.jpg"), lr_upscaled)
                    cv2.imwrite(os.path.join(save_dir, f"{base_name}_hr.jpg"), img)
        
        print(f"Created low-res/high-res pairs in {save_dir}")

if __name__ == "__main__":
    preprocessor = GloveDataPreprocessor(Config())
    print("Preprocessing utilities ready!")
'''
    with open('utils/preprocessing.py', 'w') as f:
        f.write(preprocessing_code)

    print("\\n✅ All files created successfully!")
    print("\\n🎯 Next Steps:")
    print("1. Place your dataset in 'data/raw/' folder")
    print("2. Run: python run_system.py")
    print("3. Choose option 1 to train SRGAN")
    print("4. Then choose option 2 to train YOLO")
    print("5. Finally choose option 3 to run web interface")

if __name__ == "__main__":
    create_all_files()