import torch
import torch.nn as nn

class VectorQuantiser(nn.Module):
    def __init__(self, embedding_num, embedding_dim):
        super(VectorQuantiser, self).__init__()

        self._embeddings = nn.Embedding(num_embeddings=embedding_num, embedding_dim=embedding_dim)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel=2, pool_stride=2):
        super(DownSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride) # This halves the height and width.
        )

    def forward(self, inputs):
        return self._block(inputs)

class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        # Input is (B, C=1, W=64, H=64)
        self._encoder_conv = nn.Sequential(
            DownSampleBlock(1, 32, 3, 1, 1), # (B, C=32, W=32, H=32)
            DownSampleBlock(32, 64, 3, 1, 1),  # (B, C=64, W=16, H=16)
            DownSampleBlock(64, 128, 3, 1, 1),  # (B, C=128, W=8, H=8)
            DownSampleBlock(128, 256, 3, 1, 1),  # (B, C=256, W=4, H=4)
            nn.Flatten(1), # Change to vector, not an image. (B, C*W*H = 4096)

            nn.Linear(4096, 4 * 4 * embedding_dim),

            # Reshape to (B, C=embedding_dim, W=4, H=4)
            nn.Unflatten(1, (embedding_dim, 4, 4))
        )

    def forward(self, inputs):
        return self._encoder_conv(inputs)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(UpSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

    def forward(self, inputs):
        return self._block(inputs)

class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()

        self._decoder_block = nn.Sequential(
            # Input is (B, C=embedding_dim, W=4, H=4)

            nn.Flatten(1),  # Change to vector.
            nn.Linear(embedding_dim * 4 * 4, 4096),
            nn.Unflatten(1, (256 * 4 * 4)), # (B, C=256, W=4, H=4)

            UpSampleBlock(256, 128, 3, 2, 1, 1), # (B, C=128, W=8, H=8)
            UpSampleBlock(128, 64, 3, 2, 1, 1),  # (B, C=64, W=16, H=16)
            UpSampleBlock(64, 32, 3, 2, 1, 1), # (B, C=32, W=32, H=32)

            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), # (B, C=1, W=64, H=64)
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self._decoder_block(inputs)

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        embedding_dim = 64
        embedding_num = 10000

        self._encoder = Encoder(embedding_dim)
        self._quantiser = VectorQuantiser(embedding_num, embedding_dim)
        self._decoder = Decoder(embedding_dim)
