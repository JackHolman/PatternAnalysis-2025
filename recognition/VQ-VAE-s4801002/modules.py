import torch
import torch.nn as nn

class VectorQuantiser(nn.Module):
    """
    Module that takes input vectors and quantises them to the nearest vectors in the codebook.
    Codebook vectors are initially assigned uniformly over [-1/embedding_num, 1/embedding_num], however through
    the codebook loss component returned by the forward pass they can be updated to more similar to the input.

    The structure of this class is based on an example by Kashif Rasul [2].
    """

    def __init__(self, embedding_num: int, embedding_dim: int) -> None:
        super(VectorQuantiser, self).__init__()

        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim

        # Create codebook embeddings.
        self._embeddings = nn.Embedding(embedding_num, embedding_dim)

        # Distribute embeddings uniformly between 1/embedding_num and -1/embedding_num
        self._embeddings.weight.data.uniform_(-1/embedding_num, 1/embedding_num)

    def forward(self, inputs: torch.Tensor, disp_unique: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quantiser. Returns quantised version of inputs along with commitment loss
        (MSE between inputs and quantised output with quantised detached, designed to pull inputs closer to assigned
        codebook vectors) and codebook loss (MSE between inputs and quantised with inputs detached, designed to
        pull codebook vectors closer to the inputs assigned to them).

        :param inputs: Input tensor of shape (B, embedding_dim, W, H).
        :param disp_unique: True to print the number of unique codebook indices assigned to inputs, False otherwise.
        :return: (quantised inputs, commitment loss, codebook loss)
        """

        # Input is (B, C=self._embedding_dim, W=4, H=8)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs_flattened = inputs.view(-1, self._embedding_dim)

        # Calculate the distance between each of the input vectors and each vector in the codebook.
        flat_sq_inputs = torch.sum(inputs_flattened**2, dim=1, keepdim=True)
        sq_weights = torch.sum(self._embeddings.weight**2, dim=1)
        product = torch.matmul(inputs_flattened, self._embeddings.weight.t())

        distances = flat_sq_inputs + sq_weights - 2 * product

        # Now get encoding numbers.
        encoding_indices = torch.argmin(distances, dim=1)

        # Print the number of unique codes used in this quantisation, if enabled.
        if disp_unique:
            with torch.no_grad():
                unique_codes = torch.unique(encoding_indices).numel()
                print(f"Codes used: {unique_codes}/{self._embedding_num}")

        # Convert encoding indices to a one hot matrix that will copy the encoding vectors out when multiplied
        # with the codebook.
        encodings = torch.nn.functional.one_hot(encoding_indices, self._embedding_num).type(inputs.dtype)

        # Quantisation. Multiply one-hot encodings with the codebook embeddings.
        quantised = torch.matmul(encodings, self._embeddings.weight).view(inputs.shape)

        # Loss
        commitment_loss = nn.functional.mse_loss(quantised.detach(), inputs)
        codebook_loss = nn.functional.mse_loss(quantised, inputs.detach())

        # Straight through estimator
        quantised = inputs + (quantised - inputs).detach()

        return quantised.permute(0, 3, 1, 2).contiguous(), commitment_loss, codebook_loss


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection and ReLU activation. Shape of output is equal to shape of input.
    """

    def __init__(self, channels: int) -> None:
        """
        Initialise residual block.
        :param channels: The number of channels for input and output Tensors.
        """

        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        :param inputs: Tensor of input.
        :return: Tensor of output being the sum of the input tensor and the output of the convolution block.
        """
        return self._block(inputs) + inputs


class DownSampleBlock(nn.Module):
    """
    2D Convolutional downsample block. Takes input of shape (B, in_channels, W, H) and returns output of
    shape (B, out_channels, W/2, H/2).

    Instance Normalises and applies LeakyReLU before output.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=4, stride: int=2, padding: int=1) -> None:
        """
        Initialise downsample block.
        :param in_channels: Number of channels in input tensor.
        :param out_channels: Number of channels in output tensor.
        :param kernel_size: The size of the kernel used in convolutional downsample. Default = 4.
        :param stride: Stride of the kernel during convolution. Default = 2.
        :param padding: Padding used during convolution. Default = 1.
        """
        super(DownSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through downsample block.
        :param inputs: Input tensor.
        :return: Block output tensor.
        """
        return self._block(inputs)



class Encoder(nn.Module):
    """
    Encoder module, takes input shaped (B, 1, 64, 128) and outputs shape (B, embedding_dim, 4, 8).
    Applies 4 convolutional downsamples followed by residual blocks, then downsizes the channels of the output into
    the embedding dimension.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Initialise encoder. Designed for input of shape (B, C=1, W=64, H=128).
        Designed to output shape (B, C=embedding_dim, W=4, H=8).
        :param embedding_dim: The size of the embedding dimension to shape outputs in.
        """
        super(Encoder, self).__init__()

        # Input is (B, C=1, W=64, H=128)
        self._block = nn.Sequential(
            DownSampleBlock(1, 32), # (B, C=32, W=32, H=64)
            ResidualBlock(32), # Dimensionality unchanged.

            DownSampleBlock(32, 64),  # (B, C=64, W=16, H=32)
            ResidualBlock(64), # Dimensionality unchanged.

            DownSampleBlock(64, 128),  # (B, C=128, W=8, H=16)
            ResidualBlock(128), # Dimensionality unchanged.

            DownSampleBlock(128, 256),  # (B, C=256, W=4, H=8)
            ResidualBlock(256), # Dimensionality unchanged.

            # Now downsize channel into embedding dimension.
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0), # (B, C=embedding_dim, W=4, H=8)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        :param inputs: Tensor of shape (B, C=1, W=64, H=128)
        :return: Tensor of shape (B, C=embedding_dim, W=4, H=8)
        """
        return self._block(inputs)

class UpSampleBlock(nn.Module):
    """
    2D Convolutional transpose upsample block. Takes input of shape (B, in_channels, W, H) and returns output of
    shape (B, out_channels, W*2, H*2).

    Instance Normalises and applies LeakyReLU before output.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=4, stride: int=2, padding: int=1,
                 output_padding: int=0) -> None:
        """
        Initialise upsample block.
        :param in_channels: Number of channels in input tensor.
        :param out_channels: Number of channels in output tensor.
        :param kernel_size: The size of the kernel used in convolutional transpose upsample. Default = 4.
        :param stride: Stride of the kernel during convolution. Default = 2.
        :param padding: Padding used during convolution. Default = 1.
        :param output_padding: Output padding used during convolution. Default = 0.
        """
        super(UpSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through upsample block.
        :param inputs: Input tensor.
        :return: Block output tensor.
        """
        return self._block(inputs)

class Decoder(nn.Module):
    """
    Decoder module, takes input shaped (B, embedding_dim, 4, 8) and outputs shape (B, 1, 64, 128).

    Upsizes channels from embedding_dim to 256, then passes input through 4 blocks of residual and upsample block pairs.
    Finally, applies Sigmoid activation.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Initialise decoder. Designed for input of shape (B, C=embedding_dim, W=4, H=8).
        Designed to output shape (B, C=1, W=64, H=128).
        :param embedding_dim: The size of the embedding dimension.
        """
        super(Decoder, self).__init__()

        self._decoder_block = nn.Sequential(
            # Upsample channels out of the embedding dimension.
            nn.ConvTranspose2d(in_channels=embedding_dim, out_channels=256, kernel_size=1), # (B, C=256, W=4, H=8)
            ResidualBlock(256), # Dimensionality unchanged.

            UpSampleBlock(256, 128),  # (B, C=128, W=8, H=16)
            ResidualBlock(128), # Dimensionality unchanged.

            UpSampleBlock(128, 64), # (B, C=64, W=16, H=32)
            ResidualBlock(64), # Dimensionality unchanged.

            UpSampleBlock(64, 32),  # (B, C=32, W=32, H=64)
            ResidualBlock(32), # Dimensionality unchanged.

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0), # (B, C=1, W=64, H=128)

            # Squeeze to [0, 1].
            nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        :param inputs: Tensor of shape (B, C=embedding_dim, W=4, H=8).
        :return: Tensor of shape (B, C=1, W=64, H=128).
        """
        return self._decoder_block(inputs)

class VQVAE(nn.Module):
    """
    Vector Quantisation Variation Auto Encoder module.

    Encodes input into latent space using Encoder module.
    Quantises latent space used Vector Quantiser module.
    Decodes quantised encoding using Encoder module back into input space.
    """

    def __init__(self, embedding_dim=64, embedding_num=1024) -> None:
        """
        Initialise VQ-VAE model.
        :param embedding_dim: The dimension of the embedding space. Each embedding vector in the codebook will be of
        this dimension.
        :param embedding_num: The number of embeddings in the codebook.
        """

        super(VQVAE, self).__init__()
        self._embedding_dim = embedding_dim
        self._embedding_num = embedding_num

        self._encoder = Encoder(self._embedding_dim)
        self._quantiser = VectorQuantiser(self._embedding_num, self._embedding_dim)
        self._decoder = Decoder(self._embedding_dim)

    def forward(self, inputs: torch.Tensor, quantise=True, disp_unique=False) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE, takes input and compresses into latent space with embedding_dim dimensions,
        quantises embeddings, then attempts to reconstruct original inputs.
        :param inputs: Input tensor of shape (B, 1, 64, 128).
        :param quantise: Use the quantiser. Default True.
        :param disp_unique: Display the number of unique encodings used by the quantiser. Has no effect when
        quantise is False. Default False.
        :return: (Output tensor of shape (B, 1, 64, 128), commitment loss, codebook loss)
        """

        x = self._encoder(inputs)

        # If quantisation is disabled, do not use the quantiser and return 0 loss for commitment and codebook.
        if not quantise:
            quantised = x
            commitment_loss = torch.tensor(0.0, device=inputs.device)
            codebook_loss = torch.tensor(0.0, device=inputs.device)
        else:
            quantised, commitment_loss, codebook_loss = self._quantiser(x, disp_unique)

        out = self._decoder(quantised)

        return out, commitment_loss, codebook_loss
