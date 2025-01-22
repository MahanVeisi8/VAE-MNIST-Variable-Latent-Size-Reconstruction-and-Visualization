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

### **Key Differences: Standard AE vs VAE**

| Feature                          | **Standard Autoencoder (AE)**                                     | **Variational Autoencoder (VAE)**                            |
|----------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| **Mapping Type**                 | Deterministic: Encodes data to a fixed latent representation.     | Probabilistic: Encodes data into a latent distribution (mean and variance). |
| **Sampling**                     | No Sampling: Reconstructs outputs directly from deterministic encodings. | Supports Sampling: Enables generation of new data points by sampling from the latent space. |
| **Latent Space Regularization**  | None: Focuses on reconstruction accuracy only.                   | Uses **KL Divergence** to enforce smoothness and continuity in the latent space. |
| **Generative Capabilities**      | Limited: Cannot generate new data samples.                       | Powerful: Can generate diverse and realistic data samples. |
| **Focus**                        | Solely on reconstruction of input data.                          | Balances reconstruction and latent space organization for generative tasks. |

### **Why Probabilistic Latent Space?**

1. **Data Generation**: Sampling from a latent distribution allows VAEs to generate diverse outputs that resemble the training data.
2. **Continuity**: Nearby points in the latent space correspond to similar outputs, ensuring smooth transitions in generated data.
3. **Regularization**: The additional KL divergence term ensures the learned latent distribution aligns with a standard Gaussian, making the space interpretable and well-structured.

---

## **Core Concepts**



### **Mathematical Framework**

1. **Maximizing Data Likelihood**  
   The primary goal of a Variational Autoencoder is to maximize the likelihood of the observed data p(x). This is expressed as:

   ![Maximizing Data Likelihood](assets/MaximizingDataLikelihood.png)
   ![Integral ](assets/integral.png)

   However, solving this integral is **intractable** because integrating over all possible \( z \) is computationally expensive.

---

2. **Bayes' Rule Approximation**  
   To address this, Bayes' rule is applied:
   ![Maximizing Data Likelihood](assets/Bayes.png)

   But there’s a new problem: computing p_theta(z|x) is still challenging because it involves knowledge of the posterior, which is also intractable.

---

3. **Neural Network as an Estimator**  
   To approximate p_theta(z|x), we use a neural network q_phi(z|x) to act as the posterior. This is referred to as the variational posterior and makes the computation feasible.
   ![Reparameterization Trick](assets/reparameterizationTrick.png)

   Now, instead of directly computing the likelihood p(x), the focus shifts to maximizing a lower bound called the **Evidence Lower Bound (ELBO)**.

---

4. **Decomposing the ELBO**  
   Using the new approximation, the logarithm of p(x) can be rewritten as:

   ![Reparameterization Trick](assets/Decomposing.png)
   
   Here:
   - **ELBO**: Evidence Lower Bound, which we aim to maximize during training.
   - D_KL: Kullback-Leibler divergence between q_phi(z|x) and the true posterior p_theta(z|x).

   Since D_KL >= 0, maximizing the ELBO brings us closer to the true log-likelihood p(x).

---
5. **KL Divergence Loss**

The **Kullback-Leibler (KL) Divergence** measures the difference between the learned latent distribution q_phi(z|x) (produced by the encoder) and the prior distribution p(z) (usually a standard Gaussian N(0, 1):

   ![KL Div](assets/KL_Div.png)


This is a statistical measure to ensure the generated latent space distribution aligns closely with the desired prior distribution.

### **Why is KL Divergence Important?**

1. **Latent Space Regularization**: Ensures that the latent space is smooth, continuous, and well-organized, making it easier to sample meaningful latent vectors.
2. **Avoiding Overfitting**: Without the KL term, the latent space may overfit the training data, losing generalization to new, unseen inputs.
3. **Generative Capabilities**: A structured latent space ensures that new samples generated from the prior distribution resemble the training data.

By enforcing this regularization, KL divergence encourages the model to learn a meaningful and generative latent space.

---

6. **Final Loss Function**  
   Combining these, the VAE goal is to maximizing the lower bound:

   ![two terms](assets/two_terms.png)


   Where the first term encourages accurate reconstruction of input data and the second term regularizes the latent space to align with a standard Gaussian prior.

#### **Acknowledgments** 

The mathematical explanations and formula illustrations in this section were adapted from [Justin Johnson's EECS 498-007: Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/). Credit goes to the original author for the insightful material and visualizations.
---

### **Encoder and Decoder Roles**

1. **Encoder**:
   - Maps input data \(x\) to a latent distribution characterized by:
     - Mean (\(\mu\)) 
     - Variance (\(\sigma^2\))
   - Outputs the parameters for sampling latent representations \(z \sim \mathcal{N}(\mu, \sigma^2)\).

2. **Decoder**:
   - Maps sampled latent vectors \(z\) back to the data space.
   - Outputs the reconstructed data \(\hat{x}\).

---

### **Sampling with Reparameterization Trick**

Since direct backpropagation through stochastic sampling is not feasible, VAEs employ the **reparameterization trick**:

\[
z = \mu + \sigma \cdot \epsilon
\]

Where \(\epsilon \sim \mathcal{N}(0, 1)\) is a random noise vector. This enables gradient-based optimization while allowing latent space sampling.

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

