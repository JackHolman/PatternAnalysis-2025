import torch
import torch.nn as nn

class VectorQuantiser(nn.Module):
    def __init__(self, embedding_num: int, embedding_dim: int):
        super(VectorQuantiser, self).__init__()

        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim

        #self._embeddings = nn.Embedding(num_embeddings=embedding_num, embedding_dim=embedding_dim)
        self._embeddings = nn.Parameter(torch.randn((embedding_num, embedding_dim)))
        # We will use the default normal initialisation for the embeddings.

    def forward(self, inputs, print_usage):
        # Input is (B, C=self._embedding_dim, W=4, H=4)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        inputs_flattened = inputs.view(-1, self._embedding_dim)
        # Inputs flattened shape is (B * W * H, C=self._embedding_dim)?

        # self._embeddings.weight is (self._embedding_num, self._embedding_dim)
        # For each batch, and for each of the 4*4 encoding vectors, we need to calculate the distance to each embedding
        # vector.

        flat_sq_inputs = torch.sum(inputs_flattened**2, dim=1, keepdim=True)
        sq_weights = torch.sum(self._embeddings**2, dim=1)
        product = torch.matmul(inputs_flattened, self._embeddings.t())

        distances =  flat_sq_inputs + sq_weights - 2 * product

        # Now get encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._embedding_num, device=inputs.device)
        # I think this just grabs ones and stores them so when we multiply by the encoding weights it grabs them.
        encodings.scatter(1, encoding_indices, 1)

        if print_usage:
            unique = torch.unique(encoding_indices)
            usage = len(unique) / self._embedding_num
            print("Usage:", usage)

        # Quantise and unflatten
        quantised = torch.matmul(encodings, self._embeddings).view(inputs.shape)

        # Loss
        commitment_loss = nn.functional.mse_loss(quantised.detach(), inputs)
        codebook_loss = nn.functional.mse_loss(quantised, inputs.detach())

        # Straight through estimator
        quantised = inputs + (quantised - inputs).detach()

        return quantised.permute(0, 3, 1, 2).contiguous(), commitment_loss, codebook_loss


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

            nn.Linear(4096, 4 * 4 * embedding_dim, bias=True),

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
        )

    def forward(self, inputs):
        return self._block(inputs)

class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()

        self._decoder_block = nn.Sequential(
            # Input is (B, C=embedding_dim, W=4, H=4)

            nn.Flatten(1),  # Change to vector.
            nn.Linear(embedding_dim * 4 * 4, 4096, bias=True),
            nn.Unflatten(1, (256, 4, 4)), # (B, C=256, W=4, H=4)

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
        embedding_dim = 128
        embedding_num = 512

        self._encoder = Encoder(embedding_dim)
        self._quantiser = VectorQuantiser(embedding_num, embedding_dim)
        self._decoder = Decoder(embedding_dim)

    def forward(self, inputs, print_usage):
        x = self._encoder(inputs)
        quantised, commitment_loss, codebook_loss = self._quantiser(x, print_usage)
        out = self._decoder(quantised)

        return out, commitment_loss, codebook_loss