import logging
import gc
import random
import numpy as np
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union
from collections import defaultdict

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, get_dtype

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def get_num_transfer_tokens_maskgit(mask_index, steps, mode="linear"):
    """
    Calculate number of tokens to transfer at each step based on scheduling mode.
    """
    num_masked_tokens = mask_index.sum(dim=-1, keepdim=True).float()

    if mode == "linear":
        ratio = torch.linspace(0, 1, steps + 1, device=mask_index.device)[1:]
    elif mode == "cosine":
        t = torch.linspace(0, 1, steps + 1, device=mask_index.device)[1:]
        ratio = 1 - torch.cos(t * np.pi / 2)
    elif mode == "pow2":
        t = torch.linspace(0, 1, steps + 1, device=mask_index.device)[1:]
        ratio = t ** 2
    elif mode == "sqrt":
        t = torch.linspace(0, 1, steps + 1, device=mask_index.device)[1:]
        ratio = torch.sqrt(t)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    num_transfer_tokens = (ratio.unsqueeze(0) * num_masked_tokens).round().long()
    num_transfer_tokens = torch.diff(num_transfer_tokens, dim=-1, prepend=torch.zeros_like(num_transfer_tokens[:, :1]))

    return num_transfer_tokens


@torch.no_grad()
def llada_rcr_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                       cfg_scale=0., remasking='low_confidence', mask_id=126336,
                       rcr=True, conf_alg='llada', mode='linear', top_p=None, top_k=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        rcr: Whether to use Running Confidence Remasking.
        conf_alg: Confidence algorithm ('llada', 'entropy', 'topk_margin', 'random').
        mode: Scheduling mode ('linear', 'cosine', 'pow2', 'sqrt').
        top_p: Top-p sampling threshold.
        top_k: Top-k sampling threshold.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    attn_mask = torch.ones_like(x)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # RCR tracking
    overtime_confidence = torch.zeros_like(x, dtype=torch.float32)

    for num_block in range(num_blocks):
        start_idx = prompt.shape[1] + num_block * block_length
        end_idx = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block, mode=mode)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            # Get model predictions
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_, attention_mask=torch.cat([attn_mask, attn_mask], dim=0)).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attn_mask).logits

            if not rcr:
                # Vanilla LLaDA sampling when RCR is disabled
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    confidence = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                elif remasking == 'random':
                    confidence = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)
            else:
                # RCR sampling logic
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature

                # Apply top_p and top_k filtering
                if top_p is not None and top_p < 1:
                    logits = top_p_logits(logits, top_p)
                if top_k is not None:
                    logits = top_k_logits(logits, top_k)

                # Handle confidence algorithm
                if conf_alg == 'random':
                    p = torch.rand(x.shape, device=x.device)
                else:
                    p = F.softmax(logits, dim=-1)

                # Sample x0 and compute confidence
                if temperature > 0:
                    try:
                        x0 = torch.distributions.Categorical(probs=p).sample()
                        confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    except:
                        confidence, x0 = p.max(dim=-1)
                else:
                    confidence, x0 = p.max(dim=-1)

                # Apply specific confidence algorithms
                if conf_alg == 'entropy':
                    epsilon = 1e-8
                    log_probs = torch.log(p + epsilon)
                    confidence = -torch.sum(p * log_probs, dim=-1)  # Negative entropy (higher is more confident)
                elif conf_alg == 'topk_margin':
                    sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                    top1_probs = sorted_probs[:, :, 0]
                    top2_probs = sorted_probs[:, :, 1]
                    confidence = top1_probs - top2_probs

            if not rcr:
                # Vanilla LLaDA token selection logic
                confidence[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, confidence, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            else:
                # RCR token selection logic
                # Ensure we don't process tokens beyond the current block
                confidence[:, end_idx:] = -np.inf

                # Update predictions for masked positions only
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()

                    # RCR: Select tokens based on cumulative confidence
                    total_remaining_tokens = num_transfer_tokens[j, i:].sum().item()
                    _, select_indices = torch.topk(confidence[j], k=total_remaining_tokens)
                    x[j, select_indices] = x0[j, select_indices]
                    overtime_confidence[j, select_indices] = confidence[j, select_indices].clone().float()

                    # RCR: Re-mask lowest confidence tokens for next steps
                    if i != (steps_per_block - 1):
                        # Only consider tokens in the current generation block for remasking
                        current_block_conf = overtime_confidence[j, start_idx:end_idx].clone()
                        current_block_conf_wo_zeros = torch.where(
                            current_block_conf == 0.0, 1.0, current_block_conf
                        )
                        num_tokens_to_mask = num_transfer_tokens[j, i + 1:].sum().item()
                        if num_tokens_to_mask > 0 and len(current_block_conf_wo_zeros) > 0:
                            _, local_mask_indices = torch.topk(
                                current_block_conf_wo_zeros,
                                k=min(num_tokens_to_mask, len(current_block_conf_wo_zeros)),
                                largest=False
                            )
                            # Convert local indices to global indices
                            global_mask_indices = local_mask_indices + start_idx
                            x[j, global_mask_indices] = mask_id

    return x


def empty_cache_by_memory(threshold_gb=70):
    """
    Empty CUDA cache if allocated memory exceeds threshold
    Args:
        threshold_gb: Memory threshold in GB
    """
    if torch.cuda.is_available():
        # Get current memory allocated
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB

        if allocated > threshold_gb:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cache cleared. Memory freed: {allocated:.2f} GB")


@register_model("llada_rcr")
class LLaDA_RCR(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_prompt_len: Optional[int] = 1024,
        steps: Optional[int] = 128,
        gen_length: Optional[int] = 128,
        block_length: Optional[int] = 128,
        temperature: Optional[float] = 0.0,
        cfg_scale: Optional[float] = 0.0,
        remasking: Optional[str] = 'low_confidence',
        mask_id: Optional[int] = 126336,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        # RCR specific parameters
        rcr: Optional[bool] = True,
        conf_alg: Optional[str] = 'llada',
        mode: Optional[str] = 'linear',
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        # generation params
        self.max_prompt_len = max_prompt_len
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.mask_id = mask_id

        # RCR specific params
        self.rcr = rcr
        self.conf_alg = conf_alg
        self.mode = mode
        self.top_p = top_p
        self.top_k = top_k

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        self.model = (
            transformers.AutoModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config: Optional[dict] = None
    ) -> "LLaDA_RCR":
        """
        Constructor from optional string representation of arguments.

        Args:
        - arg_string (str): String representation of arguments (e.g. "key1=value1,key2=value2").
        - additional_config (dict): Additional configuration to pass to the constructor.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        # tokenize
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        prompt_ids = prompt_ids[:, -self.max_prompt_len:]
        prompt_ids = prompt_ids.to(device=self.device)

        # generate using RCR-enabled LLaDA algorithm
        responses = []
        for i, prompt_tensor in enumerate(prompt_ids):
            # LLaDA requires single prompt processing
            single_prompt = prompt_tensor.unsqueeze(0)  # Add batch dimension

            # Use the RCR-enabled LLaDA generation function
            generation_ids = llada_rcr_generate(
                self.model,
                single_prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id,
                rcr=self.rcr,
                conf_alg=self.conf_alg,
                mode=self.mode,
                top_p=self.top_p,
                top_k=self.top_k
            )

            # Decode the response
            response = self.tokenizer.decode(generation_ids[0, single_prompt.shape[1]:], skip_special_tokens=True)
            responses.append(response)

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)

            for i, r in enumerate(responses):
                for s in gen_args[0]['until']:
                    r = r.split(s)[0]
                responses[i] = r

            if self.rank == 0:
                print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")

            res.extend(responses)
            pbar.update(len(contexts))

            # Clear memory after each batch
            empty_cache_by_memory(threshold_gb=70)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not implemented for diffusion models"""
        raise NotImplementedError("Loglikelihood not supported for diffusion models")

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """Not implemented for diffusion models"""
        raise NotImplementedError("Loglikelihood rolling not supported for diffusion models")