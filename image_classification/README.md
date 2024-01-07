## Image classification on Dogs vs. Cats Kaggle dataset

### Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats). You can download it from the link, or use the
Kaggle API: `kaggle competitions download -c dogs-vs-cats`.

### Model

The model is a custom sequence of:

* a 4x4 convolutional layer with non-overlapping stride, which acts as a 4x HxW downsampler
* a configurable number of 3x3 residual blocks (one block is 3x3 conv, BN, GELU, 3x3 conv, BN, residual connection, GELU)
* a Vision Transformer (ViT) encoder using a prepended learnable "classification token"

The Vision Transformer is not exactly the same as the one described in the paper, as the positional embeddings are added to
the queries and keys before each attention block, instead of being added to the input embeddings before the transformer encoder.

The tweaks to the ViT architecture are inspired by DETR.

### Attention visualizations

Below are some visualizations of certain attention heads in certain layers of the ViT encoder. The more red an area is, the higher
the attention weight is for that area. The images are my own, and are not part of the dataset.

#### Layer 1, head 1
<img src="./imgs/cat_1_attention_head_1.jpeg" alt="Attention scores" width="300" height="300"/>

This head seems to focus on the cat's ears.

#### Layer 2, head 14
<img src="./imgs/cat_1_attention_head_14.png" alt="Attention scores" width="300" height="300"/>

This head seems to focus on the cat's eyes and nose.
