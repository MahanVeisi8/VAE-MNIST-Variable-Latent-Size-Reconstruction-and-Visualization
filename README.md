# **Variational Autoencoders (VAE) for MNIST Dataset**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M9R-see6buUaLwAm2bcJuihywgW14ebw?usp=sharing)  
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/release/python-380/)  
[![Status](https://img.shields.io/badge/status-active-green)]()

Welcome to this project exploring **Variational Autoencoders (VAE)** on the **MNIST** dataset! 🚀

This repository demonstrates training VAEs with different latent sizes (2, 4, and 16) and showcases the impact of latent space dimensions through:

- **Reconstruction Performance**
- **Latent Space Visualizations**
- **Training Metrics**

![Latent Space](assets/latent_16/test_latent_visualization.gif)

---

## **Overview**

Variational Autoencoders (VAEs) extend traditional Autoencoders (AEs) by introducing a **probabilistic framework** to the latent space. This enhancement provides better generalization, continuity, and the ability to generate new data samples.

Key distinctions include:

### **Standard Autoencoder**
- Maps input data deterministically to a fixed latent representation.
- Lacks the ability to generate new samples or capture data variability.

### **Variational Autoencoder (VAE)**
- Models the latent space as a **probability distribution**, typically Gaussian.
- Enables sampling from the latent space for data generation.
- Includes a **KL Divergence** term in the loss to regularize the latent space.

Here is an illustrative comparison:

![AE vs VAE](assets/autoencoder_vs_vae.jpg)

The following sections explore the VAE structure, its mathematical formulation, and experimental results on the MNIST dataset.

---

## **Core Concepts**

### **Latent Space in VAE**
In a standard AE, the encoder maps input data to a fixed latent vector. In contrast, VAE encodes inputs into a probabilistic distribution, allowing the latent space to:

1. **Encourage Continuity**: Similar inputs yield close latent encodings.
2. **Enable Sampling**: Random samples from the distribution generate diverse and realistic outputs.

Key advantages of this design:
- Smoother latent space.
- Improved generalization.
- Capability to generate new, meaningful data.

### **Mathematical Formulation**

The VAE loss combines two terms:

1. **Reconstruction Loss**
   Measures how accurately the model reconstructs the input data:

   \[
   \mathcal{L}_{Recon} = \text{MSE}(x, \hat{x})
   \]

   or Binary Cross-Entropy (BCE) for binary inputs:

   \[
   \mathcal{L}_{Recon} = -\mathbb{E}_{q(z|x)} \left[\log p(x|z)\right]
   \]

2. **KL Divergence Loss**
   Regularizes the latent space by enforcing similarity between the learned distribution \( q(z|x) \) and a standard Gaussian prior \( p(z) \):

   \[
   \mathcal{L}_{KL} = D_{KL}\left(q(z|x) \parallel p(z)\right)
   \]

   KL Divergence is given by:

   \[
   D_{KL}(P \parallel Q) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right)
   \]

The total VAE loss is:

![loss](assets/loss.jpg)


\[
\mathcal{L}_{VAE} = \mathcal{L}_{Recon} + \beta \cdot \mathcal{L}_{KL}
\]

Where \( \beta \) controls the weight of the KL term.

![KL Divergence](assets/kl_divergence_visual.png)

---

## **Model Architecture**

### **Encoder**
Maps input images \( x \) to a latent distribution \( q(z|x) \) characterized by:
- Mean (\( \mu \))
- Variance (\( \sigma^2 \))

### **Latent Sampling**
To enable backpropagation through the probabilistic latent space, VAE employs the **reparameterization trick**:

\[
 z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
\]

### **Decoder**
Generates reconstructions \( \hat{x} \) from sampled latent vectors \( z \).

![VAE Architecture](assets/vae_architecture.png)

---

## **Experimental Results**

### **Latent Space Visualization**
Below are the latent space visualizations for different latent sizes. The GIFs illustrate how the latent space evolves during training.

<table>
  <tr>
    <th>Latent Size</th>
    <th>Latent 2</th>
    <th>Latent 4</th>
    <th>Latent 16</th>
  </tr>
  <tr>
    <td>Latent Space Visualization</td>
    <td><img src="assets/latent_2/test_latent_visualization.gif" alt="Latent Space 2" width="350"></td>
    <td><img src="assets/latent_4/test_latent_visualization.gif" alt="Latent Space 4" width="350"></td>
    <td><img src="assets/latent_16/test_latent_visualization.gif" alt="Latent Space 16" width="350"></td>
  </tr>
</table>

### **Reconstruction Results**

Sample reconstructions of test images at various latent sizes:

<table>
  <tr>
    <th>Latent Size</th>
    <th>Reconstruction GIF</th>
  </tr>
  <tr>
    <td align="center">2</td>
    <td><img src="assets/latent_2/sample_test_reconstruction.gif" alt="Reconstruction Latent 2" width="400"></td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td><img src="assets/latent_4/sample_test_reconstruction.gif" alt="Reconstruction Latent 4" width="400"></td>
  </tr>
  <tr>
    <td align="center">16</td>
    <td><img src="assets/latent_16/sample_test_reconstruction.gif" alt="Reconstruction Latent 16" width="400"></td>
  </tr>
</table>

---

## **Metrics Tracking**

Training metrics, including **MSE**, **SSIM**, and **PSNR**, were tracked for train and test sets. Here are the plots for each latent size:

### Latent Size: 2
![Metrics train Latent 2](assets/latent_2/train_plots.png)
![Metrics test Latent 2](assets/latent_2/test_plots.png)

### Latent Size: 4
![Metrics train Latent 4](assets/latent_4/train_plots.png)
![Metrics test Latent 4](assets/latent_4/test_plots.png)

### Latent Size: 16
![Metrics train Latent 16](assets/latent_16/train_plots.png)
![Metrics test Latent 16](assets/latent_16/test_plots.png)

---

## **References and Credits**

1. **Images and Concepts**: Thanks to authors and resources such as [source 1](#) and [source 2](#) for providing visualizations.
2. **Math Formulas**: KL Divergence explanations were adapted from authoritative ML texts.

---

### 🚀 Happy Experimenting!

