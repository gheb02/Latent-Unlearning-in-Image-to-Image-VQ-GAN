# Latent Unlearning in Image-to-Image VQ-GAN

## Abstract

High-fidelity vector-quantized generative models such as VQ-GAN pose safety challenges due to their ability to reproduce explicit visual content. This work investigates Latent Code Replacement (LCR), a training-free unlearning strategy that operates directly on the discrete latent space by identifying and substituting codebook entries associated with unwanted content. Through a frequency-based analysis across explicit and safe datasets, we show that in high-resolution image-to-image settings explicit information is not localized in individual codes but diffused across combinations of indices. As a consequence, global code replacement strategies require large-scale substitutions and induce severe image-wide artifacts. To address this limitation, we propose Localized LCR, which constrains replacement to spatial regions detected as explicit using an external classifier. Experimental results demonstrate that localized random substitution effectively suppresses explicit features while largely preserving overall image quality, whereas semantically motivated nearest-neighbor replacements fail. Further analysis reveals that VQ-GAN codes lack independent semantic meaning and that the decoder enforces only local consistency, explaining the ineffectiveness of semantic code shifts. These findings highlight fundamental limitations of code-level unlearning in VQ-GANs and motivate future work on patch-level and co-occurrence-based latent interventions.


*Warning: This research involves the analysis of explicit datasets; however, all sensitive visual content presented in this paper has been blurred or obscured to ensure the safety of the reader.*

## Repository Structure



## Acknowledgements

This work builds upon several excellent open-source projects:

- [VQ-GAN](https://github.com/CompVis/taming-transformers)
- [NudeNet](https://github.com/notAI-tech/NudeNet)

We thank the authors for making their code and data publicly available.
