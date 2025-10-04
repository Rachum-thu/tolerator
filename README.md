# Test-Time Token-Level Cross-Validation for dLLMs

<div align="center">
  <img src="https://github.com/Rachum-thu/tolerator/blob/main/assets/overview.jpg" alt="overview" width="100%"/>
</div>


It is often claimed that vanilla discrete diffusion language models have an ***inherent*** ability for ***error correction***, since each token is repeatedly predicted as context evolves.

However, this is ***incomplete***: once a token is ***accepted***, it becomes ***fixed*** and cannot be revised. Models like [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) leave early mistakes to ***persist*** and ***propagate*** through later steps.

To address this, we propose ***Tolerator*** — a test-time **to**ken-**le**vel c**r**oss-v**a**lida**t**i**o**n **r**efinement strategy. By decoupling diffusion decoding into sequence fill-up and refinement, it allows previously fixed tokens to be revisited and corrected, yielding robust improvements on reasoning, QA, and code generation benchmarks.

## Links
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inference Example](#inference-example)
- [Batch Evaluation](#batch-evaluation)
- [Experiment Logs](#experiment-logs)
- [License](#license)

## Installation

Create a clean folder and download the anonymous code snapshot.
```bash
mkdir tolerator && cd tolerator
wget https://anonymous.4open.science/api/repo/Tolerator-85C5/zip -O tolerator.zip
unzip tolerator.zip && rm tolerator.zip
```

Create and activate a clean environment.
```bash
conda create -n tolerator python=3.10
conda activate tolerator
```

Install dependencies.
```bash
pip install -r requirements.txt
```

Set your Hugging Face token for official model access.
```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

## Quick Start

## Inference Example

## Batch Evaluation

## Experiment Logs

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).