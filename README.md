# Test-Time Token-Level Cross-Validation for dLLMs

<div align="center">
  <img src="https://github.com/Rachum-thu/tolerator/blob/main/assets/overview.jpg" alt="overview" width="100%"/>
</div>

It is often claimed that vanilla discrete diffusion language models have an ***inherent*** ability for ***error correction***, since each token is repeatedly predicted as context evolves.

However, this is ***incomplete***: once a token is ***accepted***, it becomes ***fixed*** and cannot be revised. Models like [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) leave early mistakes to ***persist*** and ***propagate*** through later steps.

To address this, we propose ***Tolerator*** — a test-time **to**ken-**le**vel c**r**oss-v**a**lida**t**i**o**n **r**efinement strategy. By decoupling diffusion decoding into sequence fill-up and refinement, it allows previously fixed tokens to be revisited and corrected, yielding robust improvements on reasoning, QA, and code generation benchmarks.


