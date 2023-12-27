## Sentiment classification on the IMDB dataset

The model is a vanilla Transformer encoder. For classification, a special "classification token" is prepended to each text sequence,
and the decoder uses the corresponding final embedding to produce the classification result.

Corresponding Weights & Biases project [link](https://wandb.ai/balazs-opra/sentiment-classification-transformer).
