# Changes tested

In this repo we experimented with two changes to the [RoBERTa](https://arxiv.org/abs/1907.11692) baseline:

## Attention with Linear Biases (ALiBi)
Proposed in [Press et al., 2022](https://arxiv.org/abs/2108.12409), ALiBi uses linear biases to the attention weights instead of positional encoding to represent positions of the tokens. Since masked LMs like RoBERTa both look ahead and back to determine the masked tokens, considerations must be made regarding how to distinguish them (https://github.com/ofirpress/attention_with_linear_biases/issues/5). Unless otherwise noted, our implementation achieves this by [shifting the linear biases ahead by `0.5 * slope`](https://github.com/ofirpress/attention_with_linear_biases/issues/5#issuecomment-1213410982), i.e. the constant bias (right) of Figure 3 in [Press et al., 2022](https://arxiv.org/abs/2108.12409) becomes
```
 0 -.5 -1.5 -2.5 -3.5
-1   0  -.5 -1.5 -2.5
-2  -1    0  -.5 -1.5
-3  -2   -1    0  -.5
-4  -3   -2   -1    0
```

## Contrastive Language Pretraining (CLAP) head
Inspired by [CLIP](https://github.com/openai/CLIP) but actually goes back all the way to the origin of weight tying ([Press and Wolf, 2017](https://arxiv.org/abs/1608.05859), CLAP head is the simplest possible prediction head for the missing token except the thermodynamic beta (inverse temperature):
https://github.com/EIFY/fairseq/blob/8143446dfa88d9f8e246b366bd335f6c9b018db0/fairseq/models/roberta/model.py#L527-L543
Compared to the baseline prediction head, we removed the `embed_dim x embed_dim` fully-connected layer, activation function (GELU), layer norm, and the `output_dim` trainable bias. On the other hand, we added the trainable thermodynamic beta and L2-normalize the embeddings before computing the inner products between the transformer output and the embeddings, scaled by beta.

# Experiments

## WikiText-103
At first we tested the changes with the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/), using the validation set MLM perplexity as the metric. We tested the baseline, learned-clap (baseline + CLAP head), ALiBi (baseline + ALiBI), and zero-clap (baseline + CLAP head + ALiBi), in addition to baseline but with sinusoidal positional encoding:

![valid_ppl_cleaned](https://user-images.githubusercontent.com/2584418/194671509-ea3eea59-5796-433c-8986-2038f31d5c04.png)
