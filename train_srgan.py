import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from models.srgan.model import Generator, Discriminator
from utils.preprocessing import GloveDataPreprocessor
from utils.config import Config

class SRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = Generator(scale_factor=config.SCALE_FACTOR).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Loss functions
        self.adversarial_criterion = nn.BCELoss()
        self.content_criterion = nn.L1Loss()
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config.LEARNING_RATE)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.LEARNING_RATE)
        
        # Create directories
        os.makedirs(config.SRGAN_MODEL_DIR, exist_ok=True)
    
    def load_dataset(self, data_dir):
        """Load low-res and high-res image pairs"""
        lr_images = []
        hr_images = []
        
        for file in os.listdir(data_dir):
            if file.endswith('_lr.jpg'):
                lr_path = os.path.join(data_dir, file)
                hr_path = os.path.join(data_dir, file.replace('_lr.jpg', '_hr.jpg'))
                
                if os.path.exists(hr_path):
                    # Load and preprocess images
                    lr_img = cv2.imread(lr_path)
                    hr_img = cv2.imread(hr_path)
                    
                    if lr_img is not None and hr_img is not None:
                        # Convert BGR to RGB and normalize
                        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                        
                        lr_images.append(lr_img)
                        hr_images.append(hr_img)
        
        return lr_images, hr_images
    
    def train(self, train_dir, valid_dir, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        # Load data
        print("Loading training data...")
        lr_train, hr_train = self.load_dataset(train_dir)
        lr_valid, hr_valid = self.load_dataset(valid_dir)
        
        print(f"Training samples: {len(lr_train)}")
        print(f"Validation samples: {len(lr_valid)}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()
            
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for i in range(0, len(lr_train), self.config.BATCH_SIZE):
                # Get batch
                batch_lr = lr_train[i:i+self.config.BATCH_SIZE]
                batch_hr = hr_train[i:i+self.config.BATCH_SIZE]
                
                if not batch_lr:
                    continue
                
                # Convert to tensors
                lr_tensor = torch.tensor(np.array(batch_lr)).float().permute(0, 3, 1, 2).to(self.device) / 255.0
                hr_tensor = torch.tensor(np.array(batch_hr)).float().permute(0, 3, 1, 2).to(self.device) / 255.0
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                real_outputs = self.discriminator(hr_tensor)
                real_loss = self.adversarial_criterion(real_outputs, torch.ones_like(real_outputs))
                
                fake_images = self.generator(lr_tensor)
                fake_outputs = self.discriminator(fake_images.detach())
                fake_loss = self.adversarial_criterion(fake_outputs, torch.zeros_like(fake_outputs))
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                fake_outputs = self.discriminator(fake_images)
                adversarial_loss = self.adversarial_criterion(fake_outputs, torch.ones_like(fake_outputs))
                content_loss = self.content_criterion(fake_images, hr_tensor)
                
                g_loss = adversarial_loss + 0.001 * content_loss
                g_loss.backward()
                self.g_optimizer.step()
                
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'G_Loss: {g_loss_epoch/len(lr_train):.4f}, '
                      f'D_Loss: {d_loss_epoch/len(lr_train):.4f}')
            
            # Save model checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_model(epoch + 1)
    
    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, os.path.join(self.config.SRGAN_MODEL_DIR, f'srgan_epoch_{epoch}.pth'))
        
        print(f"Model saved at epoch {epoch}")

def main():
    config = Config()
    
    # Preprocess data
    preprocessor = GloveDataPreprocessor(config)
    
    # Use absolute paths from config
    train_dir = os.path.join(config.HAND_GLOVES_DIR, "train")
    valid_dir = os.path.join(config.HAND_GLOVES_DIR, "valid")
    
    # Prepare SRGAN dataset
    srgan_train_dir, srgan_valid_dir = preprocessor.prepare_srgan_dataset(train_dir, valid_dir)
    
    # Train SRGAN
    trainer = SRGANTrainer(config)
    trainer.train(srgan_train_dir, srgan_valid_dir)

if __name__ == "__main__":
    main()