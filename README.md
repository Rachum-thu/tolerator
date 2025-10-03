# Test-Time Token-Level Cross-Validation for dLLMs

<div align="center">
  <img src="https://github.com/Rachum-thu/tolerator/blob/main/assets/overview.jpg" alt="overview" width="100%"/>
</div>

It is often claimed that vanilla discrete diffusion language models possess an ***inherent*** ability for ***error correction***, since the token at each position is repeatedly predicted as the context evolves over iterations.

However, this view is ***incomplete***: once a token is accepted, it becomes ***fixed*** and cannot be revised. For example, [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) decide at every iteration whether a token should be further remasked; if it is not, the token is considered ***accepted*** and remains ***unchanged*** thereafter. As a result, any early mistake will ****persist** and **propagate*** through subsequent steps.

To overcome this, we propose ***Tolerator*** — a test-time **to**ken-**le**vel c**r**oss-v**a**lida**t**i**o**n **r**efinement strategy that decouples diffusion decoding into two stages: sequence fill-up and refinement. In refinement, tokens alternate roles between predictor and context, enabling previously fixed tokens to be revisited and corrected. This training-free approach transforms decoding into a more robust, self-validating process that consistently improves reasoning, QA, and code generation benchmarks.
