# Vector Quantisation Variational Auto Encoder

## Description of Algorithm
What is a VQ-VAE? Who proposed it? What makes it distinct? What does it hope to address?

## Description of Problem
Description of data?

### Pre-processing
The data was pre-processed by resizing and cropping each image to 128x64 before normalising the data to be in range [0, 1]. The data was resized to reduce model size and complexity, allowing for faster training convergence.  The data was normalised to [0, 1] so that a sigmoid activation function could be used as the final step in the decoder.
## How it Works
Description of the training algorithm? 
- why only use test set for eval

### Running Guide
#### Dependencies
The Python packages required for all scripts are listed in `requirements.txt`. They can be installed by running
```bash
pip3 install -r requirements.txt
```

#### Data
All scripts assume they will be run on the Rangpur cluster, so the dataset paths point to the location of the data on Rangpur. 

#### Training
In an environment with the dependencies installed, the training algorithm can be run using the following command.
```bash
python3 train.py
```

#### Inference
In an environment with the dependencies installed, a saved model can be loaded and used for inference. The `predict.py` script assumes there is a file in the current working directory named `vq-vae.model`, this is the model that will be used for inference. If this model file is present, the predict script can be run using the following command.
```bash
python3 predict.py
```

## Results
- Reconstruction loss on train set
- SSIM on train set

A plot showing each type of loss for every batch while training can be found in Figure 3.

At the conclusion of training, the model was used to produce reconstructions on the test set in inference mode. The reconstructions can be found in Figure 2. The average Structural Similarity Index Measure on the test set was 0.6111, and the average reconstruction loss was 0.0589.

### Figures
#### Figure 1: Train Set Reconstructions After 100 Epochs
![Figure 1: Train Set Reconstructions After 100 Epochs](imgs/train-100.png)
#### Figure 2: Test Set Reconstructions After Conclusion Of Training
![Figure 2: Test Set Reconstructions After Conclusion Of Training](imgs/test.png)
#### Figure 3: Plot Of Loss Of Each Training Batch
![Figure 3: Plot Of Loss Of Each Training Batch](imgs/loss.png)

## Reproducibility
Before the model starts training, seeds for inbuilt random and PyTorch random are set to a pre-determined value. Additionally, PyTorch is set to use deterministic algorithms to ensure that results are reproducible.

## References
[1] K. Rasul, “vq-vae.ipynb - Colab,” Google Colab, Nov. 14, 2019.
    https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    (accessed Oct. 29, 2025).