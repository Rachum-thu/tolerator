#!/usr/bin/env bash

export HF_ALLOW_CODE_EVAL="1"  # Required for code evaluation tasks like mbpp

# Default values
MODEL_NAME="Tolerator-org/Dream-v0-Instruct-7B-Tolerator"
TASKS="humaneval_instruct"
# Generation parameters
STEPS=32
MAX_NEW_TOKENS=256
TEMPERATURE=0.5
DENOISE_TEMP=0.3
TOP_P=0.9
ALG="entropy"
# Penalty parameters
REPETITION_PENALTY=1.0
FILLUP_EOT_PENALTY=1.2
DENOISE_EOT_PENALTY=1.0
# Denoising parameters (random_sample algorithm)
DENOISE_ALG="random_sample"
DENOISE_MAX_ITER=32
DENOISE_START_REMASK_RATIO=0.8
DENOISE_END_REMASK_RATIO=0.4
DENOISE_DECAY_SCHEDULE="cosine"
DENOISE_UPDATE_RATE=1.0
DENOISE_REPEAT_CONVERGE_THRESHOLD=10
# Evaluation parameters
LIMIT=32  # for quick test
BATCH_SIZE=1
NUM_FEWSHOT=0
SEED=42
PORT=12334
DEVICE="cuda"
OUTPUT_DIR="../output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --denoise_temp)
      DENOISE_TEMP="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --steps|--diffusion_steps)
      STEPS="$2"
      shift 2
      ;;
    --max_new_tokens|--max_tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --alg|--algorithm)
      ALG="$2"
      shift 2
      ;;
    --repetition_penalty|--penalty)
      REPETITION_PENALTY="$2"
      shift 2
      ;;
    --fillup_eot_penalty)
      FILLUP_EOT_PENALTY="$2"
      shift 2
      ;;
    --denoise_eot_penalty)
      DENOISE_EOT_PENALTY="$2"
      shift 2
      ;;
    --eot_penalty)
      # For backward compatibility, set both penalties to the same value
      FILLUP_EOT_PENALTY="$2"
      DENOISE_EOT_PENALTY="$2"
      shift 2
      ;;
    --denoise_alg)
      DENOISE_ALG="$2"
      shift 2
      ;;
    --denoise_max_iter)
      DENOISE_MAX_ITER="$2"
      shift 2
      ;;
    --denoise_start_remask_ratio)
      DENOISE_START_REMASK_RATIO="$2"
      shift 2
      ;;
    --denoise_end_remask_ratio)
      DENOISE_END_REMASK_RATIO="$2"
      shift 2
      ;;
    --denoise_decay_schedule)
      DENOISE_DECAY_SCHEDULE="$2"
      shift 2
      ;;
    --denoise_update_rate)
      DENOISE_UPDATE_RATE="$2"
      shift 2
      ;;
    --denoise_repeat_converge_threshold)
      DENOISE_REPEAT_CONVERGE_THRESHOLD="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_fewshot)
      NUM_FEWSHOT="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "Dream Evaluation Script"
      echo ""
      echo "Model Options:"
      echo "  --model MODEL_NAME          Model to evaluate (default: $MODEL_NAME)"
      echo "  --tasks TASKS               Tasks to evaluate (default: $TASKS)"
      echo ""
      echo "Generation Options:"
      echo "  --steps|--diffusion_steps STEPS  Diffusion steps (default: $STEPS)"
      echo "  --max_new_tokens|--max_tokens TOKENS  Max new tokens (default: $MAX_NEW_TOKENS)"
      echo "  --temperature TEMP          Temperature for fillup sampling (default: $TEMPERATURE)"
      echo "  --denoise_temp TEMP         Temperature for denoise sampling (default: $DENOISE_TEMP)"
      echo "  --top_p TOP_P              Top-p for sampling (default: $TOP_P)"
      echo "  --alg|--algorithm ALG       Algorithm to use (default: $ALG)"
      echo ""
      echo "Penalty Options:"
      echo "  --repetition_penalty|--penalty PENALTY  Repetition penalty (default: $REPETITION_PENALTY)"
      echo "  --fillup_eot_penalty PENALTY    EOT token penalty for fillup phase (default: $FILLUP_EOT_PENALTY)"
      echo "  --denoise_eot_penalty PENALTY   EOT token penalty for denoise phase (default: $DENOISE_EOT_PENALTY)"
      echo "  --eot_penalty PENALTY           Set both EOT penalties to same value (for compatibility)"
      echo ""
      echo "Denoising Options (random_sample algorithm):"
      echo "  --denoise_alg ALG               Denoising algorithm (default: $DENOISE_ALG)"
      echo "  --denoise_max_iter ITER         Maximum denoising iterations (default: $DENOISE_MAX_ITER)"
      echo "  --denoise_start_remask_ratio RATIO  Starting remask ratio for annealing (default: $DENOISE_START_REMASK_RATIO)"
      echo "  --denoise_end_remask_ratio RATIO    Ending remask ratio for annealing (default: $DENOISE_END_REMASK_RATIO)"
      echo "  --denoise_decay_schedule SCHEDULE  Annealing schedule: cosine or exponential (default: $DENOISE_DECAY_SCHEDULE)"
      echo "  --denoise_update_rate RATE      Update rate for denoising (default: $DENOISE_UPDATE_RATE)"
      echo "  --denoise_repeat_converge_threshold THR  Convergence threshold for position locking (default: $DENOISE_REPEAT_CONVERGE_THRESHOLD)"
      echo ""
      echo "Evaluation Options:"
      echo "  --limit LIMIT              Limit number of examples (default: $LIMIT)"
      echo "  --batch_size SIZE          Batch size (default: $BATCH_SIZE)"
      echo "  --num_fewshot NUM          Number of few-shot examples (default: $NUM_FEWSHOT)"
      echo "  --port PORT                Main process port (default: $PORT)"
      echo "  --device DEVICE            Device to use (default: $DEVICE)"
      echo "  --output_dir DIR           Output directory (default: $OUTPUT_DIR)"
      echo "  --seed SEED                Random seed for reproducibility (default: $SEED)"
      echo "  -h|--help                  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "==== Running Dream-Tolerator evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$MAX_NEW_TOKENS, temperature=$TEMPERATURE, denoise_temp=$DENOISE_TEMP, alg=$ALG"
echo "Tolerator: fillup_eot=$FILLUP_EOT_PENALTY, denoise_eot=$DENOISE_EOT_PENALTY, max_iter=$DENOISE_MAX_ITER, ratio=$DENOISE_START_REMASK_RATIO->$DENOISE_END_REMASK_RATIO, converge_threshold=$DENOISE_REPEAT_CONVERGE_THRESHOLD"
echo "Seed: $SEED"
echo "===================================="

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model diffllm \
  --model_args "pretrained=$MODEL_NAME,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,diffusion_steps=${STEPS},dtype=bfloat16,temperature=$TEMPERATURE,top_p=$TOP_P,alg=$ALG,repetition_penalty=$REPETITION_PENALTY,fillup_eot_penalty=$FILLUP_EOT_PENALTY,denoise_eot_penalty=$DENOISE_EOT_PENALTY,denoise_alg=$DENOISE_ALG,denoise_max_iter=$DENOISE_MAX_ITER,denoise_start_remask_ratio=$DENOISE_START_REMASK_RATIO,denoise_end_remask_ratio=$DENOISE_END_REMASK_RATIO,denoise_decay_schedule=$DENOISE_DECAY_SCHEDULE,denoise_update_rate=$DENOISE_UPDATE_RATE,denoise_decode_temp=$DENOISE_TEMP,denoise_repeat_converge_threshold=$DENOISE_REPEAT_CONVERGE_THRESHOLD" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEWSHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${TASKS}_dream_tolerator_steps_${STEPS}_tokens_${MAX_NEW_TOKENS}_temp_${TEMPERATURE}_fillup_eot_${FILLUP_EOT_PENALTY}_denoise_eot_${DENOISE_EOT_PENALTY}_iter_${DENOISE_MAX_ITER}_converge_${DENOISE_REPEAT_CONVERGE_THRESHOLD}_seed_${SEED}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template
