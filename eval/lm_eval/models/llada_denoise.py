import logging
import gc
import random
import numpy as np
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union

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


def apply_eot_penalty(logits, eot_penalty=1.0, eot_token_id=126081):
    """
    Apply penalty to EOT tokens only.
    Uses LLaDA's correct EOT token ID: 126081
    """
    if eot_penalty == 1.0:
        return logits
    
    # Check if the token ID is within vocab range
    vocab_size = logits.size(-1)
    if eot_token_id >= vocab_size:
        eval_logger.warning(f"EOT token ID {eot_token_id} is out of bounds for vocab size {vocab_size}. Skipping EOT penalty.")
        return logits
    
    logits[..., eot_token_id] = logits[..., eot_token_id] / eot_penalty
    return logits


def sample_tokens(logits, temperature=0.0, neg_entropy=False):
    """
    Sample tokens from logits with optional temperature and entropy-based confidence
    """
    if temperature > 0:
        logits = logits / temperature
    
    probs = torch.softmax(logits, dim=-1)
    
    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@torch.no_grad()
def llada_generate_with_eot_penalty(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                                    cfg_scale=0., remasking='low_confidence', mask_id=126336, fillup_eot_penalty=1.0):
    '''
    LLaDA generation function with EOT penalty support
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Apply EOT penalty in fillup generation
            if fillup_eot_penalty != 1.0:
                logits = apply_eot_penalty(logits, fillup_eot_penalty)
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def create_llada_denoise_func(denoise_params, prompt_length, mask_id=126336, denoise_eot_penalty=1.0):
    """
    Create a denoising function that exactly matches Dream's random_sample logic
    IMPORTANT: For LLaDA, we don't need logits shifting unlike Dream
    """
    # Extract parameters exactly like Dream
    start_rr = denoise_params.get("start_remask_ratio", None)
    end_rr = denoise_params.get("end_remask_ratio", None)
    schedule_type = denoise_params.get("decay_schedule", "cosine")
    
    iter_counter = {"i": 0}  # Exactly like Dream
    
    # Convergence tracking (exactly like Dream)
    repeat_converge_threshold = denoise_params.get("repeat_converge_threshold", None)
    locked_mask = None            # shape [gen_len] boolean – True means position is locked
    last_token = None             # shape [gen_len] long – last token observed when updated
    consec_count = None           # shape [gen_len] long – consecutive identical updates counter
    
    def _compute_remask_ratio(step: int):
        """Compute remask ratio using annealing schedule"""
        if start_rr is None or end_rr is None:
            raise ValueError("start_remask_ratio and end_remask_ratio must both be set for denoising")
        
        max_iter = denoise_params["max_iter"]
        if max_iter <= 1:
            return end_rr
        
        t = step / (max_iter - 1)
        if schedule_type == "cosine":
            import math
            return end_rr + (start_rr - end_rr) * 0.5 * (1 + math.cos(math.pi * t))
        else:  # exponential
            if start_rr == 0:
                return 0.0
            ratio = (end_rr / start_rr) ** t
            return start_rr * ratio
    
    @torch.no_grad()
    def llada_random_sample_denoise_func(model, sequences):
        """One round of random-sample denoising exactly like Dream but for LLaDA"""
        
        nonlocal locked_mask, last_token, consec_count
        
        # perform one round of random-sample denoising with dynamic remask ratio
        step = iter_counter["i"]
        iter_counter["i"] += 1
        
        device = sequences.device
        B, seq_len = sequences.shape
        assert B == 1, "random_sample only supports batch size 1"
        gen_len = seq_len - prompt_length
        if gen_len <= 0:
            return sequences, True  # nothing to denoise – treat as already fixed
        
        # lazy init state tensors once we know gen_len (exactly like Dream)
        if locked_mask is None:
            locked_mask = torch.zeros(gen_len, dtype=torch.bool, device=device)
            last_token = torch.zeros(gen_len, dtype=torch.long, device=device)
            consec_count = torch.zeros(gen_len, dtype=torch.long, device=device)
        
        # ------------------ build working position sets (exactly like Dream) ------------------
        available_rel_pos = torch.nonzero(~locked_mask, as_tuple=False).squeeze(-1)
        # if everything is fixed we can early-exit
        if available_rel_pos.numel() == 0:
            return sequences, True
        
        remask_ratio = _compute_remask_ratio(step)
        sample_ratio = 1.0 - remask_ratio
        update_rate = denoise_params.get("update_rate", 1.0)
        
        decode_temp = denoise_params.get("decode_temp", 0.)
        decode_alg = denoise_params.get("decode_alg", "origin")
        
        # determine number of tokens to keep **within available positions** (key difference!)
        num_keep = int(available_rel_pos.numel() * sample_ratio)
        if num_keep <= 0:
            num_keep = 0  # allow full mask if ratio very small
        
        # select positions to keep (exactly like Dream)
        shuffled = available_rel_pos[torch.randperm(available_rel_pos.numel(), device=device)]
        keep_rel = shuffled[:num_keep]
        
        # mask out other generated tokens (only those still available) (exactly like Dream)
        mask_rel = available_rel_pos[~torch.isin(available_rel_pos, keep_rel)]
        if mask_rel.numel() == 0:
            # nothing to mask – no forward/backward cost; all_fixed depends on convergence later
            all_fixed_flag = locked_mask.all().item()
            return sequences, all_fixed_flag
        
        mask_pos = mask_rel + prompt_length
        
        seq_masked = sequences.clone()
        seq_masked[:, mask_pos] = mask_id
        
        # forward pass - ONLY DIFFERENCE: NO LOGITS SHIFTING FOR LLADA
        outputs = model(seq_masked)
        logits = outputs.logits  # Direct use, no shifting needed!
        
        # Apply EOT penalty in denoising stage if specified
        if denoise_eot_penalty != 1.0:
            logits = apply_eot_penalty(logits, denoise_eot_penalty)
        
        # logits for masked positions (exactly like Dream)
        mask_logits = logits[:, mask_pos, :]
        probs = torch.softmax(mask_logits, dim=-1)
        orig_tokens = sequences[:, mask_pos]
        confidences_pos = torch.gather(probs, 2, orig_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0)
        
        M = confidences_pos.size(0)
        num_update = max(1, int(M * update_rate))
        idxs = torch.argsort(confidences_pos)[:num_update]
        to_update_abs = mask_pos[idxs]
        to_update_rel = to_update_abs - prompt_length
        
        # sample new tokens only for selected positions (exactly like Dream)
        mask_logits_sel = mask_logits[:, idxs, :]
        flat_logits_sel = mask_logits_sel.reshape(-1, mask_logits_sel.size(-1))
        _, sampled_tokens_sel = sample_tokens(
            flat_logits_sel,
            temperature=decode_temp,
            neg_entropy=(decode_alg == "entropy")
        )
        sampled_tokens_sel = sampled_tokens_sel.reshape(1, -1)
        new_sequences = sequences.clone()
        new_sequences[:, to_update_abs] = sampled_tokens_sel
        
        # ------------------ convergence bookkeeping (exactly like Dream) ------------------
        all_fixed_flag = False
        if repeat_converge_threshold is not None and repeat_converge_threshold > 0:
            # update stats only for positions that were actually re-sampled
            updated_vals = sampled_tokens_sel.squeeze(0)  # shape [U]
            prev_same = updated_vals == last_token[to_update_rel]
            consec_count[to_update_rel] = torch.where(prev_same,
                                                      consec_count[to_update_rel] + 1,
                                                      torch.ones_like(consec_count[to_update_rel]))
            last_token[to_update_rel] = updated_vals
            # mark newly fixed positions
            newly_fixed = consec_count[to_update_rel] >= repeat_converge_threshold
            if newly_fixed.any():
                locked_mask[to_update_rel[newly_fixed]] = True
            all_fixed_flag = locked_mask.all().item()
        
        return new_sequences, all_fixed_flag
    
    return llada_random_sample_denoise_func


@torch.no_grad()
def apply_random_sample_denoise(model, sequences, prompt_length, denoise_params, mask_id=126336, denoise_eot_penalty=1.0):
    """
    Apply random sample denoising exactly like Dream's implementation
    """
    max_iter = denoise_params.get("max_iter", 10)
    
    # Validate annealing parameters
    start_rr = denoise_params.get("start_remask_ratio", None)
    end_rr = denoise_params.get("end_remask_ratio", None)
    
    if start_rr is None or end_rr is None:
        raise ValueError("Both start_remask_ratio and end_remask_ratio must be set for denoising")
    
    # Create denoise function exactly like Dream
    denoise_func = create_llada_denoise_func(denoise_params, prompt_length, mask_id, denoise_eot_penalty)
    
    # Main denoising loop exactly like Dream
    iter = 0
    while iter < max_iter:
        result_ret = denoise_func(model, sequences)
        # Support both (sequences, converged) and sequences-only return styles
        
        if isinstance(result_ret, tuple):
            sequences, is_converged = result_ret
        else:
            sequences = result_ret
            is_converged = False
        
        iter += 1
        if is_converged:
            break
    
    return sequences


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
            eval_logger.info(f"Cache cleared. Memory freed: {allocated:.2f} GB")


@register_model("llada_denoise")
class LLaDADenoise(LM):
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
        fillup_eot_penalty: Optional[float] = 1.0,
        denoise_eot_penalty: Optional[float] = 1.0,
        # Denoising parameters
        denoise_max_iter: Optional[int] = 16,
        start_remask_ratio: Optional[float] = 0.8,
        end_remask_ratio: Optional[float] = 0.4,
        decay_schedule: Optional[str] = 'cosine',
        denoise_update_rate: Optional[float] = 1.0,
        denoise_temp: Optional[float] = 0.3,
        repeat_converge_threshold: Optional[int] = 10,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
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
        self.fillup_eot_penalty = fillup_eot_penalty
        self.denoise_eot_penalty = denoise_eot_penalty
        
        # denoising params
        self.denoise_max_iter = denoise_max_iter
        self.start_remask_ratio = start_remask_ratio
        self.end_remask_ratio = end_remask_ratio
        self.decay_schedule = decay_schedule
        self.denoise_update_rate = denoise_update_rate
        self.denoise_temp = denoise_temp
        self.repeat_converge_threshold = repeat_converge_threshold

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
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

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

        # generate using LLaDA algorithm with denoising
        responses = []
        for i, prompt_tensor in enumerate(prompt_ids):
            # LLaDA requires single prompt processing
            single_prompt = prompt_tensor.unsqueeze(0)  # Add batch dimension
            
            # Use the LLaDA generation function with fillup EOT penalty
            generation_ids = llada_generate_with_eot_penalty(
                self.model,
                single_prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id,
                fillup_eot_penalty=self.fillup_eot_penalty
            )
            
            # Apply denoising
            denoise_params = {
                "max_iter": self.denoise_max_iter,
                "start_remask_ratio": self.start_remask_ratio,
                "end_remask_ratio": self.end_remask_ratio,
                "decay_schedule": self.decay_schedule,
                "update_rate": self.denoise_update_rate,
                "decode_temp": self.denoise_temp,
                "decode_alg": "origin",
                "repeat_converge_threshold": self.repeat_converge_threshold
            }
            
            generation_ids = apply_random_sample_denoise(
                self.model,
                generation_ids,
                single_prompt.shape[1],  # prompt_length
                denoise_params,
                self.mask_id,
                self.denoise_eot_penalty
            )

            # decode response (remove prompt part)
            response = self.tokenizer.decode(
                generation_ids[0, single_prompt.shape[1]:].tolist(), 
                skip_special_tokens=True
            )
            
            # Split at EOS token if present
            if self.tokenizer.eos_token:
                response = response.split(self.tokenizer.eos_token)[0]
            
            responses.append(response)

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)

            for i, r in enumerate(responses):
                for s in gen_args[0]['until']:
                    r = r.split(s)[0]
                responses[i] = r

            res.extend(responses)
            
            # Clear memory after each batch
            empty_cache_by_memory(threshold_gb=70)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # For now, return dummy values as loglikelihood computation for LLaDA is complex
        # This can be implemented later if needed for specific tasks
        eval_logger.warning("loglikelihood method not implemented for LLaDA model. Returning dummy values.")
        return [(0.0, False) for _ in requests]

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("loglikelihood_rolling not implemented for LLaDA model")