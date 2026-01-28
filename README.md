# Fill-and-Refine Decoding for Diffusion Large Language Models

<div align="center">
  <img src="https://github.com/Rachum-thu/tolerator/blob/main/assets/overview.jpg" alt="overview" width="100%"/>
</div>


It is often claimed that vanilla discrete diffusion language models have an ***inherent*** ability for ***error correction***, since each token is repeatedly predicted as context evolves.

However, this is ***incomplete***: once a token is ***accepted***, it becomes ***fixed*** and cannot be revised. Models like [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) leave early mistakes to ***persist*** and ***propagate*** through later steps.

To address this, we propose ***FiRe*** — a **Fi**ll-then-**Re**fine strategy (Tolerator is the former name). By decoupling diffusion decoding into sequence fill-up and refinement, it allows previously fixed tokens to be revisited and corrected, yielding robust improvements on reasoning, QA, and code generation benchmarks.

## Links
- [Installation](#installation)
- [Quick Start](#quick-start)
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

Navigate to the evaluation directory:
```bash
cd eval
```

### Run Tolerator (Our Method)

Evaluate **LLaDA-Tolerator**:
```bash
bash eval_llada_tolerator.sh
```

Evaluate **Dream-Tolerator**:
```bash
bash eval_dream_tolerator.sh
```

You can modify parameters in these scripts (e.g., `TASKS`, `LIMIT`, `STEPS`, `DENOISE_MAX_ITER`) to customize the evaluation.

### Run Baselines

Evaluate baseline methods for comparison:

**LLaDA baselines:**
```bash
bash eval_llada_vanilla.sh   # Vanilla LLaDA
bash eval_llada_rcr.sh        # LLaDA + RCR
bash eval_llada_remdm.sh      # LLaDA + ReMDM
```

**Dream baselines:**
```bash
bash eval_dream_vanilla.sh    # Vanilla Dream
bash eval_dream_rcr.sh        # Dream + RCR
bash eval_dream_remdm.sh      # Dream + ReMDM
```

All evaluation results will be saved to the `output/` directory.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
