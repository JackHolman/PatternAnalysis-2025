import torch
import torch.nn as nn

class VectorQuantiser(nn.Module):
    def __init__(self):
        super(VectorQuantiser, self).__init__()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()

        self._encoder = Encoder()
        self._quantiser = VectorQuantiser()
        self._decoder = Decoder()
