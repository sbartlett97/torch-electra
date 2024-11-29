# [WIP] ELECTRA Training Implementation

This repository contains a custom implementation of the ELECTRA training method using PyTorch and HuggingFace's Transformers library. ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) is a transformer-based model pre-trained as a discriminator rather than a generator, resulting in a more compute-efficient pre-training approach.

This repo is currently a work in progress. My aim is to create an easy to use implementation of this training method that can be extended to trial it with more than just the base models provided on huggingface/generated from the same configs.


## References

This implementation is inspired by the original ELECTRA paper and an existing PyTorch implementation:

- **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**
  Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning  
  [ICLR 2020](https://openreview.net/pdf?id=r1xMH1BtvB)  
  ```bibtex
  @inproceedings{clark2020electra,
    title = {{ELECTRA}: Pre-training Text Encoders as Discriminators Rather Than Generators},
    author = {Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
    booktitle = {ICLR},
    year = {2020},
    url = {https://openreview.net/pdf?id=r1xMH1BtvB}
  }
  ```

- **PyTorch Implementation of ELECTRA**
  Richard Wang  
  [GitHub Repository](https://github.com/richarddwang/electra_pytorch)  
  ```bibtex
  @misc{electra_pytorch,
    author = {Richard Wang},
    title = {PyTorch implementation of ELECTRA},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/richarddwang/electra_pytorch}}
  }
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the HuggingFace team and the PyTorch community for their invaluable tools and resources.
