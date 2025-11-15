import torch
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
from torchvision.transforms import ToTensor, Resize, Compose


class GloveDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()
    
    def _load_image_pairs(self):
        """Load low-res and high-res image pairs"""
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
        
        # Load images
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')
        
        # Convert to tensors
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        else:
            # Default transform
            transform = transforms.ToTensor()
            lr_image = transform(lr_image)
            hr_image = transform(hr_image)
        
        return lr_image, hr_image

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
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.999)
        )
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.psnr_scores = []
        
        # Create model directory
        os.makedirs(config.SRGAN_MODEL_DIR, exist_ok=True)
    
    def calculate_psnr(self, generated, target):
        """Calculate PSNR between generated and target images"""
        mse = torch.mean((generated - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        print(f"Starting SRGAN training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.generator.train()
            self.discriminator.train()
            
            g_loss_epoch = 0
            d_loss_epoch = 0
            psnr_epoch = 0
            
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real images
                real_outputs = self.discriminator(hr_imgs)
                real_loss = self.adversarial_criterion(real_outputs, torch.ones_like(real_outputs))
                
                # Fake images
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
                
                # Total generator loss (with content weight)
                g_loss = adversarial_loss + 0.001 * content_loss
                g_loss.backward()
                self.g_optimizer.step()
                
                # Calculate PSNR
                psnr = self.calculate_psnr(fake_imgs, hr_imgs)
                
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                psnr_epoch += psnr
                
                if batch_idx % 50 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                          f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}, PSNR: {psnr:.2f}')
            
            # Calculate epoch averages
            avg_g_loss = g_loss_epoch / len(train_loader)
            avg_d_loss = d_loss_epoch / len(train_loader)
            avg_psnr = psnr_epoch / len(train_loader)
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.psnr_scores.append(avg_psnr)
            
            # Validation phase
            val_psnr = self.validate(val_loader)
            
            # Print epoch summary
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, '
                  f'Train_PSNR: {avg_psnr:.2f}, Val_PSNR: {val_psnr:.2f}')
            
            # Save model checkpoint
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                self.save_model(epoch + 1)
                self.plot_training_progress()
    
    def validate(self, val_loader):
        """Validate the model"""
        self.generator.eval()
        total_psnr = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                fake_imgs = self.generator(lr_imgs)
                psnr = self.calculate_psnr(fake_imgs, hr_imgs)
                total_psnr += psnr
        
        return total_psnr / len(val_loader)
    
    def save_model(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'psnr_scores': self.psnr_scores
        }
        
        model_path = os.path.join(self.config.SRGAN_MODEL_DIR, f'srgan_epoch_{epoch}.pth')
        torch.save(checkpoint, model_path)
        print(f"Model saved: {model_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.g_losses)
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.d_losses)
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.psnr_scores)
        plt.title('PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.SRGAN_MODEL_DIR, 'training_progress.png')
        plt.savefig(plot_path)
        plt.close()

def main():
    config = Config()
    
    # Data transforms
    transform = Compose([
        Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        ToTensor(),
        ])
    
    # Use absolute paths from config
    train_data_dir = os.path.join(config.HAND_GLOVES_DIR, "train", "images")
    val_data_dir = os.path.join(config.HAND_GLOVES_DIR, "valid", "images")
    
    # Create datasets
    train_dataset = GloveDataset(train_data_dir, transform=transform)
    val_dataset = GloveDataset(val_data_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize and train SRGAN
    trainer = SRGANTrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()