import torch
import torch.nn as nn

class VectorQuantiser(nn.Module):
    def __init__(self, embedding_num: int, embedding_dim: int):
        super(VectorQuantiser, self).__init__()

        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim

        self._embeddings = nn.Embedding(embedding_num, embedding_dim)

        # Distribute embeddings uniformly between 1/embedding_num and -1/embedding_num
        self._embeddings.weight.data.uniform_(-1/embedding_num, 1/embedding_num)

    def forward(self, inputs, disp_unique):
        # Input is (B, C=self._embedding_dim, W=4, H=8)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs_flattened = inputs.view(-1, self._embedding_dim)

        # self._embeddings.weight is (self._embedding_num, self._embedding_dim)
        # For each batch, and for each of the 4*4 encoding vectors, we need to calculate the distance to each embedding
        # vector.
        flat_sq_inputs = torch.sum(inputs_flattened**2, dim=1, keepdim=True)
        sq_weights = torch.sum(self._embeddings.weight**2, dim=1)
        product = torch.matmul(inputs_flattened, self._embeddings.weight.t())

        distances =  flat_sq_inputs + sq_weights - 2 * product

        # Now get encodings
        encoding_indices = torch.argmin(distances, dim=1)

        if disp_unique:
            with torch.no_grad():
                unique_codes = torch.unique(encoding_indices).numel()
                print(f"Codes used: {unique_codes}/{self._embedding_num}")

        encodings = torch.nn.functional.one_hot(encoding_indices, self._embedding_num).type(inputs.dtype)

        # Quantisation.
        quantised = torch.matmul(encodings, self._embeddings.weight).view(inputs.shape)

        commitment_loss = nn.functional.mse_loss(quantised.detach(), inputs)
        codebook_loss = nn.functional.mse_loss(quantised, inputs.detach())

        # Straight through estimator
        quantised = inputs + (quantised - inputs).detach()

        return quantised.permute(0, 3, 1, 2).contiguous(), commitment_loss, codebook_loss


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, inputs):
        return self._block(inputs) + inputs


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel=2, pool_stride=2):
        super(DownSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(True),
            #nn.Dropout2d(0.2),
        )

    def forward(self, inputs):
        return self._block(inputs)



class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        # Input is (B, C=1, W=128, H=256)
        self._encoder_conv = nn.Sequential(
            DownSampleBlock(1, 32, 4, 2, 1), # (B, C=32, W=64, H=128)
            ResidualBlock(32),
            DownSampleBlock(32, 64, 4, 2, 1),  # (B, C=64, W=32, H=64)
            ResidualBlock(64),
            DownSampleBlock(64, 128, 4, 2, 1),  # (B, C=128, W=16, H=32)
            ResidualBlock(128),
            DownSampleBlock(128, 256, 4, 2, 1),  # (B, C=256, W=8, H=16)
            ResidualBlock(256),
            # Now compress into embedding space.
            nn.Conv2d(256, embedding_dim, 1, 1, 0), # (B, C=embedding_dim, W=4, H=8)
        )

    def forward(self, inputs):
        return self._encoder_conv(inputs)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(UpSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, inputs):
        return self._block(inputs)

class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()

        self._decoder_block = nn.Sequential(
            # 1x1 convolution to increase number of channels
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=1), # (B, C=512, W=4, H=8)
            ResidualBlock(256),
            UpSampleBlock(256, 128, 4, 2, 1, 0),  # (B, C=128, W=16, H=32)
            ResidualBlock(128),
            UpSampleBlock(128, 64, 4, 2, 1, 0), # (B, C=64, W=32, H=64)
            ResidualBlock(64),
            UpSampleBlock(64, 32, 4, 2, 1, 0),  # (B, C=32, W=64, H=128)
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, 0), # (B, C=1, W=128, H=256)
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self._decoder_block(inputs)

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        embedding_dim = 64
        embedding_num = 1024

        self._encoder = Encoder(embedding_dim)
        self._quantiser = VectorQuantiser(embedding_num, embedding_dim)
        self._decoder = Decoder(embedding_dim)

    def forward(self, inputs, quantise=True, disp_unique=False):
        """
        Forward pass.
        :param inputs: Input tensor
        :param quantise: Use the quantiser. Default True.
        :param disp_unique: Display the number of unique encodings produced by the quantiser. Has no effect when
        quantise is False. Default False.
        :return: (Output tensor, commitment loss, codebook loss)
        """

        x = self._encoder(inputs)

        if not quantise:
            quantised = x
            commitment_loss = torch.tensor(0.0, device=inputs.device)
            codebook_loss = torch.tensor(0.0, device=inputs.device)
        else:
            quantised, commitment_loss, codebook_loss = self._quantiser(x, disp_unique)

        out = self._decoder(quantised)

        return out, commitment_loss, codebook_loss
