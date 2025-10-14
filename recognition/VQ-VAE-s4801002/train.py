from modules import VQVAE
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
batch_size = 128
learning_rate = 0.001

train_set = torchvision.datasets.ImageFolder(root="data/keras_png_slices_data/keras_png_slices_train", transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(),
    transforms.ToTensor()
]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root="data/keras_png_slices_data/keras_png_slices_test", transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(),
    transforms.ToTensor()
]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)

if __name__ == "__main__":
    model = VQVAE().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    beta = 0.25 # commitment loss scalar.

    # train
    for epoch in range(100):
        model.train()

        for (batch_idx, (images, _)) in enumerate(train_loader):
            model.train()
            images = images.to(device)

            # forward pass
            recon, commitment_loss, codebook_loss = model(images)

            # calculate full loss.
            recon_loss = nn.functional.mse_loss(recon, images)
            loss = (4 * recon_loss) + codebook_loss + (beta * commitment_loss)
            loss.backward()
            optimiser.step()

            if batch_idx % 20 == 0:
                # print status
                print(f"Epoch {epoch} Batch {batch_idx}: \t Loss: {loss.item():.4f} \t Recon Loss: {2*recon_loss.item():.4f} \t Codebook Loss: {codebook_loss.item():.4f} \t Commitment Loss: {beta*commitment_loss.item():.4f}")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_images, _ = next(iter(test_loader))
                test_images = test_images.to(device)
                recon_images, _, _ = model(test_images)

                # Plot original vs reconstructed
                fig, axes = plt.subplots(2, 10, figsize=(20, 4))
                for i in range(10):
                    # Original
                    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_ylabel('Original', fontsize=12)

                    # Reconstructed
                    axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_ylabel('Reconstructed', fontsize=12)

                plt.suptitle(f'VQ-VAE Reconstructions (Test Set)', fontsize=14)
                plt.tight_layout()
                #plt.show()
                plt.savefig(f"plots/test-{epoch}.png")

                test_images, _ = next(iter(train_loader))
                test_images = test_images.to(device)
                recon_images, _, _ = model(test_images)

                # Plot original vs reconstructed
                fig, axes = plt.subplots(2, 10, figsize=(20, 4))
                for i in range(10):
                    # Original
                    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_ylabel('Original', fontsize=12)

                    # Reconstructed
                    axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_ylabel('Reconstructed', fontsize=12)

                plt.suptitle(f'VQ-VAE Reconstructions (Train Set)', fontsize=14)
                plt.tight_layout()
                #plt.show()
                plt.savefig(f"plots/train-{epoch}.png")

