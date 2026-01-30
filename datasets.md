# Benchmark Datasets

A collection of datasets comparable to MNIST and dSprites for machine learning research.

## MNIST-like Datasets

| Dataset | Description | Size | Resolution |
|---------|-------------|------|------------|
| **MNIST** | Handwritten digits (0-9) | 70,000 images | 28x28 grayscale |
| **Fashion-MNIST** | Clothing items (t-shirts, pants, etc.) - drop-in replacement for MNIST | 70,000 images | 28x28 grayscale |
| **MedMNIST** | Biomedical image collection (12 2D + 6 3D datasets) | ~708K 2D, ~10K 3D | 28x28 / 28x28x28 |
| **CIFAR-10** | Natural images in 10 classes | 60,000 images | 32x32 color |
| **CIFAR-100** | Natural images in 100 classes | 60,000 images | 32x32 color |
| **EMNIST** | Extended MNIST with handwritten letters and digits | 814,255 images | 28x28 grayscale |
| **KMNIST** | Kuzushiji-MNIST - Japanese hiragana characters | 70,000 images | 28x28 grayscale |

## dSprites-like Disentanglement Datasets

| Dataset | Description | Factors | Size |
|---------|-------------|---------|------|
| **dSprites** | 2D shapes with 6 ground truth latent factors (color, shape, scale, rotation, x, y) | 6 factors | 737,280 images |
| **Shapes3D** | 3D shapes with factors: floor color, wall color, object color, size, shape, azimuth | 6 factors | 480,000 images (64x64 RGB) |
| **dSprites-Scream** | dSprites with textured backgrounds for added visual complexity | 6 factors | Variable |
| **Infinite dSprites (idSprites)** | Procedurally generates unlimited 2D shapes for continual learning | Configurable | Unlimited |
| **3D Chairs** | 3D rendered chairs with varying pose and style | Pose, style | ~86,000 images |
| **CelebA** | Face images with 40 binary attributes | 40 attributes | 202,599 images |
| **dMelodies** | Audio disentanglement benchmark with musical melodies | 9 factors | 1,524,096 samples |

## Resources & Links

### MNIST-like
- [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)
- [MedMNIST](https://medmnist.com/)
- [List of MNIST-like Datasets](https://www.simonwenkel.com/lists/datasets/list-of-mnist-like-datasets.html)
- [Keras Built-in Datasets](https://keras.io/api/datasets/)

### Disentanglement
- [dSprites - DeepMind GitHub](https://github.com/google-deepmind/dsprites-dataset)
- [dSprites on TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/dsprites)
- [dSprites on Hugging Face](https://huggingface.co/datasets/randall-lab/dsprites)
- [dSprites Variants - DeepWiki](https://deepwiki.com/google-research/disentanglement_lib/5.1-dsprites-and-variants)
- [Infinite dSprites Paper](https://arxiv.org/html/2312.16731v3)

### General
- [25 Open Datasets for Deep Learning](https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/)
- [Best Public Datasets for ML 2025](https://365datascience.com/trending/public-datasets-machine-learning/)
