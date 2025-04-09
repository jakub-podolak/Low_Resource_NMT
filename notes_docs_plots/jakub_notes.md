## Papers Overview

**Sennrich, 2016, Neural Machine Translation of Rare Words with Subword Units**
- They use BPE for NMT so the model can generate weird words by combining smaller (subword) elements. Their dictionary is more compact, and their models yield better results.

**Edunov, 2018, Understanding Back-Translation at Scale**
- They use Transformer to back-translate samples, and they observed that greedy / beam decoding in back-translation produces easy training samples, resulting in lower BLEU gains than with noisy beam or sampling from the BT model's distribution.

**Burlot, 2018, Using Monolingual Data in Neural Machine Translation: a Systematic Study**
- Back-translation is only worth its computational cost when good quality can be achieved. If the budget is limited or the model is bad it's better to use simpler heuristics like target copies with noise, optionally enhanced by GANs.
