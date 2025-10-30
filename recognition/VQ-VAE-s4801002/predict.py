import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import HipMRIStudyDataset, HipMRIStudyTransforms
from modules import VQVAE
from train import plot_sample_vs_reconstructions

# Device configuration ========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')


# Synthetic images ============================================================
def plot_synthetic(imgs: torch.Tensor, title: str, file_name: str, num_samples: int=5) -> None:
    """
    Plot a batch of synthetic images.
    :param imgs: Tensor containing synthetic images.
    :param title: The plot title.
    :param file_name: The name of the file to save the plot to.
    :param num_samples: The number of samples to plot.
    """

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        # Plot image
        axes[i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    # Save
    plt.savefig(file_name)

    # Clean up
    plt.close()

def generate(model: VQVAE, samples: int=5) -> None:
    """
    Generates a batch of fully synthetic images using random encodings.

    Since the encoding selection is random, the generated images don't have the human structure the dataset images have,
    however they do have a similar flesh texture.

    :param model: VQ-VAE model to use to generate images.
    :param samples: The number of images to generate. Default 5.
    """
    print(f"Generating and plotting {samples} synthetic images.")

    # Create a random embedding, before quantisation.
    pre_quant_embed = torch.randn((samples, 64, 8, 4)).to(device) * 3

    # Quantise the embeddings.
    quantised, _, _ = model._quantiser(pre_quant_embed, disp_unique=True)

    # Decode quantised embeddings.
    decoded = model._decoder(quantised)

    # Now plot synthetic images.
    plot_synthetic(decoded, "Synthetic images", "synthetic.png", num_samples=samples)


# Inference on a dataset. =====================================================
def infer_dataset(model: VQVAE) -> None:
    """
    Run inference on all images in a dataset. Plot a sample against reconstructions.
    Calculates the average reconstruction loss and average SSIM and outputs.
    :param model: VQVAE model to use.
    """
    print("Running inference on dataset")

    # Define the data set to use.
    inference_set = HipMRIStudyDataset("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate",
                                   transform=HipMRIStudyTransforms)
    inference_loader = torch.utils.data.DataLoader(inference_set, batch_size=16, shuffle=False)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_ssim = 0.0
    num_batches = 0
    total_recon_loss = 0.0

    for batch_idx, imgs in enumerate(inference_loader):
        imgs = imgs.to(device)
        recon_images, _, _ = model(imgs)

        # Accumulate batch SSIM.
        total_ssim += ssim(recon_images, imgs)

        # Accumulate batch reconstruction loss.
        total_recon_loss += nn.functional.l1_loss(recon_images, imgs)

        if num_batches == 0:
            plt_title = f'Sample Of VQ-VAE Reconstructions Batch SSIM={total_ssim:.4f}'
            file_name = "inference.png"
            plot_sample_vs_reconstructions(imgs, recon_images, plt_title, file_name)

        num_batches += 1

    avg_ssim = total_ssim / num_batches
    print(f"Average inference SSIM: {avg_ssim:.4f}")

    avg_loss = total_recon_loss / num_batches
    print(f"Average inference reconstruction loss: {avg_loss:.4f}")

# =============================================================================
if __name__ == "__main__":
    model = torch.load("vq-vae.model", weights_only=False).to(device)
    model.eval()

    with torch.no_grad():
        # Generate synthetic images using the quantiser and decoder.
        generate(model)

        # Take images and encode, quantise, decode. Plot original against reconstructions. Outputs average
        # reconstruction loss and average SSIM.
        infer_dataset(model)