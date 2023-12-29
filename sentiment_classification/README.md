## Sentiment classification on the IMDB dataset

The model is a vanilla Transformer encoder. For classification, a special "classification token" is prepended to each text sequence,
and the decoder uses the corresponding final embedding to produce the classification result.

Corresponding Weights & Biases project [link](https://wandb.ai/balazs-opra/sentiment-classification-transformer).

### Wandb sweeps

To use [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps):

1. Set the parameters in the [./config/wandb_sweep.yaml](./config/wandb_sweep.yaml) first. The possible parameter values are based on
what's available in the Hydra config file [./config/train_config.yaml](./config/train_config.yaml).
1. From the root of the repository, run `wandb sweep sentiment_classification/config/wandb_sweep.yaml`. This will print out the command
you need to run next.
1. Start the Sweep Agent using the command from the previous step.

### Example results

Visualization of the attention scores from the second layer of the Transformer encoder for all heads. A darker color means higher attention.

![Attention scores](./imgs/attention_viz.png)
