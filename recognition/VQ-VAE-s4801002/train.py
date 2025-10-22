from modules import VQVAE
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from dataset import HipMRIStudyDataset
from normalise import Normalise

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
manualSeed = 20650
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Needed for reproducible results
torch.use_deterministic_algorithms(True)

batch_size = 64
learning_rate = 1e-4

# Images are 256x128
train_set = HipMRIStudyDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train", transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 128)),
    transforms.CenterCrop((256, 128)),
    Normalise() # Custom normalisation transform to get data in [0, 1]
]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = HipMRIStudyDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test", transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 128)),
    transforms.CenterCrop((256, 128)),
    Normalise() # Custom normalisation transform to get data in [0, 1]
]))

#test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    quantise = True
    model = VQVAE().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    beta = 0.2 # commitment loss scalar.
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


    # train
    for epoch in range(300):
        model.train()

        for (batch_idx, images) in enumerate(train_loader):
            model.train()
            images = images.to(device)

            # forward pass
            recon, commitment_loss, codebook_loss = model(images, quantise=quantise)
            # calculate full loss.
            recon_loss = nn.functional.l1_loss(recon, images)
            loss = recon_loss + (beta * commitment_loss) + codebook_loss
            optimiser.zero_grad() # this is important.
            loss.backward()
            optimiser.step()

            if batch_idx % 20 == 0:
                # print status
                print(f"Epoch {epoch} Batch {batch_idx}: \t Loss: {loss.item():.4f} \t Recon Loss: {recon_loss.item():.4f} \t Commitment Loss: {beta*commitment_loss.item():.4f} \t Codebook Loss: {codebook_loss.item():.4f}")

        if epoch % 5 == 0 or epoch < 5:
            model.eval()
            with torch.no_grad():
                imgs = next(iter(train_loader))
                imgs = imgs.to(device)
                recon_images, _, _ = model(imgs, quantise=quantise, disp_unique=True)

                ssim_batch_val = ssim(recon_images, imgs)
                #print(f"Batch SSIM: {ssim_batch_val}")

                # Plot original vs reconstructed
                fig, axes = plt.subplots(2, 10, figsize=(20, 4))
                for i in range(10):
                    # Original
                    axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_ylabel('Original', fontsize=12)

                    # Reconstructed
                    axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_ylabel('Reconstructed', fontsize=12)

                plt.suptitle(f'VQ-VAE Reconstructions (Train Set) SSIM={ssim_batch_val}', fontsize=14)
                plt.tight_layout()
                #plt.show()
                plt.savefig(f"plots/train-{epoch}.png")

    # save the model.
    torch.save(model, "vq-vae.model")

    # Training complete, now run on test set.
    total_test_ssim = 0
    test_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_loader):
            imgs = imgs.to(device)
            recon_images, _, _ = model(imgs, quantise=quantise)

            total_test_ssim += ssim(recon_images, imgs)

            if test_batches == 0:
                # Plot original vs reconstructed
                fig, axes = plt.subplots(2, 10, figsize=(20, 4))
                for i in range(10):
                    # Original
                    axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_ylabel('Original', fontsize=12)

                    # Reconstructed
                    axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_ylabel('Reconstructed', fontsize=12)

                plt.suptitle(f'VQ-VAE Reconstructions (Test Set) batch ssim={total_test_ssim}', fontsize=14)
                plt.tight_layout()
                plt.savefig(f"plots/test.png")


            test_batches+= 1

        avg_ssim = total_test_ssim / test_batches
        print(f"Average test SSIM={avg_ssim:.4f}")