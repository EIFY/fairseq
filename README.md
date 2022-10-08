# ALiBi and CLAP head experiments

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
Inspired by [CLIP](https://github.com/openai/CLIP) but actually goes back all the way to the origin of weight tying ([Press and Wolf, 2017](https://arxiv.org/abs/1608.05859)), CLAP head is the simplest possible prediction head for the missing token except the thermodynamic beta (inverse temperature):
https://github.com/EIFY/fairseq/blob/8143446dfa88d9f8e246b366bd335f6c9b018db0/fairseq/models/roberta/model.py#L527-L543
Compared to the baseline prediction head, we removed the `embed_dim x embed_dim` fully-connected layer, activation function (GELU), layer norm, and the `output_dim` trainable bias. On the other hand, we added the trainable thermodynamic beta and L2-normalize the embeddings before feeding them to the transformer and computing the inner products between them and the transformer output, scaled by beta.

# Experiments

## [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
At first we tested the changes with the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/), using the validation set MLM perplexity as the metric. We tested the baseline, learned-clap (baseline + CLAP head), ALiBi (baseline + ALiBi), and zero-clap (baseline + CLAP head + ALiBi), in addition to baseline but with sinusoidal positional encoding:

![valid_ppl_cleaned](https://user-images.githubusercontent.com/2584418/194671509-ea3eea59-5796-433c-8986-2038f31d5c04.png)

where solid lines are what's considered "canonical" setup and dotted lines are experiments with minor differences in setup. We tested the following and found them to be irrelevant:

1. Whether we use attention dropout or not
2. Whether we use symmetrical ALiBi (option 1 in https://github.com/ofirpress/attention_with_linear_biases/issues/5) or asymmetrical ALiBi above
3. Whether we use zero vector or a separate learnable embedding for the mask embedding
4. Whether we L2-normalize the embeddings for the CLAP head or not

As we can see, the dotted lines are almost on top of the solid lines. Notably, sinusoidal positional encoding underperforms significantly compared to the baseline.

## [The Pile](https://arxiv.org/abs/2101.00027)
As the next step, we scaled our experiments to train on [the Pile](https://arxiv.org/abs/2101.00027) for one epoch. About half of the examples in the Pile has sequence length > 1024, so we set sequence length to 2048. Even so, ~1/7 of the examples have sequence length > 2048 and had to be discarded. In the end, one epoch consists of 133082 updates and [we employ cosine learning rate schedule while "overestimating" the number of training steps by 10%](https://github.com/EIFY/fairseq/blob/33fb2c306851f104cc567b7fe865b1e3fd1e6fe7/examples/roberta/config/pretraining/baseline_pile.yaml#L31-L36), as inspired by [the Chinchilla paper](https://arxiv.org/abs/2203.15556). In addition to the validation MLM perplexity, we also fine-tuned the models on the [GLUE](https://gluebenchmark.com/) benchmark. As in the original RoBERTa paper, we tested both the `roberta.base` with 125M parameters and `roberta.large` with 355M parameters. These experiments were performed on 8 x A100 40GB SXM4 GPUs, where the `roberta.base` experiments took ~3 days and `roberta.large` experiments took ~9 days. In the table below, `PPL` is the final validation MLM perplexity, `STS-B` is the best validation loss, and all the others are the best validation accuracies over 10 epochs of finetuning.

### `roberta.base`
```
             PPL↓ CoLA MNLI MRPC QNLI QQP  RTE  SST-2 STS-B↓
baseline     2.94 83.6 84.2 90   91.6 91.3 73.6 92.1  0.028
learned-clap 2.86 81.7 84.4 86.3 90.9 91.2 72.6 92.5  0.027
alibi        2.93 69.2 85.1 80.9 92   91.5 63.9 93.1  0.033
zero-clap    2.83 70.5 84.9 75.5 90.6 91.1 54.9 89.7  0.041
```
### `roberta.large`
```
             PPL↓ CoLA MNLI MRPC QNLI QQP  RTE  SST-2 STS-B↓
baseline*    2.55 83.7 86.8 84.3 92.5 91.8 79.8 93.3  0.027
learned-clap 2.5  84.1 86.3 89.7 92.8 91.7 79.8 93.7  0.023
alibi        2.65 69.1 86.5 68.4 92.4 91.7 52.7 93.6  0.123
zero-clap    2.54 69.1 86.7 81.9 92.2 91.6 52.7 93.1  0.031
```
\**Loss spiked somewhere between 24000-24500 updates and the model failed to recover. Loosely following the practice of `5.1 Training Instability` in [the PaLM paper](https://arxiv.org/abs/2204.02311), we solved the issue by restarting the training from the 20000 updates checkpoint with the PyTorch random seed changed from `1` to `2`.*

We found that ALiBi no longer helps lowering the validation MLM perplexity. Furthermore, ALiBi turned out to be harmful for several specific GLUE tasks (`CoLA`, `MRPC`, and `RTE`). CLAP head on its own, however, seems to be competitive and in fact outperforms the baseline with `roberta.large`.

# Conclusions
This seems to be another case where models with lower perplexity do not necessarily yield higher accuracies for downstream tasks and architectural changes beneficial for models at smaller scales do not imply the same for models at larger scales ([Tay et al., 2022](https://arxiv.org/abs/2207.10551)). CLAP head, however, is simpler than the standard prediction head for MLMs, requires minimal changes, and may be worth trying especially at larger scales. 
