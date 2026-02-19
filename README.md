# Face-Modification-using-VAE-βVAE-VQVAE-and-GAN

Implemented and compared multiple generative modeling approaches — VAE, β-VAE, VQ-VAE with PixelCNN prior, and DCGAN — for face reconstruction, attribute manipulation, and realistic face synthesis using the CelebA dataset (128×128 resolution). Evaluated reconstruction quality, disentanglement, realism, and latent controllability across models trained for 10 epochs.

---

## Project Overview

This project explores how different latent representations influence:

- Reconstruction quality  
- Attribute manipulation  
- Sample realism  
- Latent space structure  
- Training stability  

Models implemented:

1. Variational Autoencoder (VAE)  
2. β-VAE (β = 2, 4, 10)  
3. VQ-VAE + PixelCNN Prior  
4. DCGAN  

All models were trained for 10 epochs on the CelebA dataset using TensorFlow with GPU acceleration.

---

## Dataset

Dataset used: CelebA (CelebFaces Attributes Dataset)

- Over 200,000 celebrity face images  
- RGB images resized to 128×128  
- Used for unsupervised generative modeling  

Images were loaded locally from Google Drive and processed using TensorFlow `tf.data` pipelines.

---

## Global Configuration

```python
IMG_SIZE = 128
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 10
```

Framework used:

```python
TensorFlow + Keras
```

---

## System Pipeline

The complete experimental pipeline consists of:

1. Dataset Loading & Preprocessing  
2. VAE Implementation  
3. β-VAE Training (β = 2, 4, 10)  
4. VQ-VAE with Discrete Latent Space  
5. PixelCNN Prior Modeling  
6. DCGAN Implementation  
7. Comparative Evaluation  

---

## 1. Dataset Preprocessing

Images are:

- Loaded using `tf.data.Dataset`
- Decoded from JPEG
- Resized to 128×128
- Normalized to range [0,1]
- Shuffled and batched

Preprocessing function:

```python
def preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0
```

---

## 2. Variational Autoencoder (VAE)

### Architecture

Encoder:
- 3 Conv2D layers
- Latent mean (z_mean)
- Latent log variance (z_log_var)

Decoder:
- Dense + Reshape
- Conv2DTranspose layers
- Sigmoid output (RGB reconstruction)

Latent sampling:

```python
z = z_mean + exp(0.5 * z_log_var) * epsilon
```

### Loss Function

Total Loss:

```
Reconstruction Loss + β * KL Divergence
```

For standard VAE:

```
β = 1
```

### Training

```python
vae = VAE(beta=1.0)
vae.compile(optimizer=Adam(1e-4))
vae.fit(celeba_ds, epochs=10)
```

---

## 3. β-VAE (Disentangled Representation Learning)

β-VAE introduces a scaling factor on the KL divergence term.

Values used:

```python
β = 2
β = 4
β = 10
```

Higher β:

- Encourages disentanglement
- Improves interpretability
- Reduces reconstruction fidelity

Training loop:

```python
for b in [2,4,10]:
    model = VAE(beta=b)
    model.fit(celeba_ds, epochs=10)
```

---

## 4. VQ-VAE (Vector Quantized VAE)

Unlike VAE, VQ-VAE uses a discrete latent space.

### Key Components

- Encoder outputs continuous embeddings  
- VectorQuantizer maps embeddings to nearest codebook vector  
- Decoder reconstructs from quantized latents  

Codebook configuration:

```python
CODEBOOK_SIZE = 512
EMBED_DIM = 64
```

Loss includes:

- Reconstruction loss  
- Codebook loss  
- Commitment loss  

Training:

```python
vqvae = VQVAE()
vqvae.compile(optimizer=Adam(2e-4))
vqvae.fit(celeba_ds, epochs=10)
```

---

## 5. PixelCNN Prior

PixelCNN is trained on discrete latent index maps from VQ-VAE.

Purpose:

- Model spatial dependency in discrete latent space  
- Enable autoregressive sampling  

Architecture:

- 6 Conv2D layers  
- Sparse categorical cross-entropy loss  

---

## 6. GAN (DCGAN)

DCGAN architecture includes:

### Generator

- Dense → Reshape (16×16×256)  
- 3 Conv2DTranspose layers  
- Output: 128×128×3 image  

### Discriminator

- 3 Conv2D layers  
- LeakyReLU activations  
- Dense output (real/fake logits)  

GAN latent dimension:

```python
GAN_LATENT_DIM = 100
```

Loss:

```python
BinaryCrossentropy(from_logits=True)
```

Training:

- Alternate generator and discriminator updates  
- 10 epochs  
- Adam optimizer (β1 = 0.5)  

---

## Comparative Analysis

The four generative models were compared across:

- Reconstruction quality  
- Latent interpolation smoothness  
- Attribute controllability  
- Realism of generated faces  
- Training stability  
- Computational complexity  

---

## VAE Results

- Successfully reconstructed faces  
- Smooth latent interpolation  
- Stable and fast training  
- Slightly blurry outputs  

Observation:  
VAE learns smooth continuous latent spaces but sacrifices sharpness.

---

## β-VAE Results (β = 2, 4, 10)

- Higher β improves disentanglement  
- Individual latent dimensions control interpretable attributes  
- Higher β reduces reconstruction quality  

Trade-off observed:

Higher β → Better control  
Higher β → Blurrier reconstruction  

Observation:  
Best model for controllable face editing.

---

## VQ-VAE + PixelCNN Results

- Sharper reconstructions than VAE  
- Discrete latent space reduces blur  
- PixelCNN improves sampling realism  
- Higher computational cost  

Observation:  
Strong balance between reconstruction and generation.

---

## GAN (DCGAN) Results

- Most realistic and sharp faces  
- High visual fidelity  
- No encoder → Cannot reconstruct specific faces  
- Training less stable  

Observation:  
Best model for pure face synthesis.

---

## Key Insights

- Continuous latent spaces → Smooth interpolation (VAE)  
- Disentangled latents → Interpretability (β-VAE)  
- Discrete representations → Sharper reconstruction (VQ-VAE)  
- Adversarial training → Highest realism (GAN)  

No single model optimizes reconstruction, control, and realism simultaneously.

Generative modeling involves trade-offs between interpretability and visual fidelity.

---

## Conclusion

From this comparative study:

- VAE is a stable and simple baseline  
- β-VAE is best for controllable attribute manipulation  
- VQ-VAE improves reconstruction sharpness using discrete codes  
- GAN produces the most photorealistic faces  

Model selection depends on task:

- Editing → β-VAE  
- Reconstruction → VQ-VAE  
- Realistic synthesis → GAN  

This project demonstrates practical understanding of generative modeling, latent space design, disentanglement, adversarial training, and probabilistic deep learning architectures.
