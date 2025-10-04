#!/usr/bin/env bash

export HF_ALLOW_CODE_EVAL="1"  # Required for code evaluation tasks like mbpp

# Default values
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
TASKS="humaneval_instruct"  # other options including mbpp_instruct, gsm8k_cot, gpqa_main_cot_zeroshot, triviaqa
# Generation parameters
STEPS=32
MAX_NEW_TOKENS=256
BLOCK_LENGTH=256
TEMPERATURE=0.5
DENOISE_TEMP=0.3
CFG_SCALE=0.0
REMASKING="low_confidence"
MASK_ID=126336
# Penalty parameters
FILLUP_EOT_PENALTY=1.2
DENOISE_EOT_PENALTY=1.0
# Denoising parameters
DENOISE_MAX_ITER=32
START_REMASK_RATIO=0.8
END_REMASK_RATIO=0.4
DECAY_SCHEDULE="cosine"
DENOISE_UPDATE_RATE=1.0
REPEAT_CONVERGE_THRESHOLD=10
# Evaluation parameters
LIMIT=32  # for quick test
BATCH_SIZE=1
NUM_FEWSHOT=0
SEED=42
PORT=12345
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
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --max_new_tokens|--gen_length)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --block_length)
      BLOCK_LENGTH="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --cfg_scale)
      CFG_SCALE="$2"
      shift 2
      ;;
    --remasking)
      REMASKING="$2"
      shift 2
      ;;
    --mask_id)
      MASK_ID="$2"
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
    --denoise_max_iter)
      DENOISE_MAX_ITER="$2"
      shift 2
      ;;
    --start_remask_ratio)
      START_REMASK_RATIO="$2"
      shift 2
      ;;
    --end_remask_ratio)
      END_REMASK_RATIO="$2"
      shift 2
      ;;
    --decay_schedule)
      DECAY_SCHEDULE="$2"
      shift 2
      ;;
    --denoise_update_rate)
      DENOISE_UPDATE_RATE="$2"
      shift 2
      ;;
    --denoise_temp)
      DENOISE_TEMP="$2"
      shift 2
      ;;
    --repeat_converge_threshold)
      REPEAT_CONVERGE_THRESHOLD="$2"
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
      echo "LLaDA with Denoising Evaluation Script"
      echo ""
      echo "Model Options:"
      echo "  --model MODEL_NAME          Model to evaluate (default: $MODEL_NAME)"
      echo "  --tasks TASKS               Tasks to evaluate (default: $TASKS)"
      echo ""
      echo "Generation Options:"
      echo "  --steps STEPS               LLaDA sampling steps (default: $STEPS)"
      echo "  --max_new_tokens|--gen_length LENGTH  Generated answer length (default: $MAX_NEW_TOKENS)"
      echo "  --block_length LENGTH       Block length for semi-autoregressive (default: $BLOCK_LENGTH)"
      echo "  --temperature TEMP          Temperature for fillup sampling (default: $TEMPERATURE)"
      echo "  --denoise_temp TEMP         Temperature for denoise sampling (default: $DENOISE_TEMP)"
      echo "  --cfg_scale SCALE           Classifier-free guidance scale (default: $CFG_SCALE)"
      echo "  --remasking STRATEGY        Remasking strategy: 'low_confidence' or 'random' (default: $REMASKING)"
      echo "  --mask_id ID                Mask token ID (default: $MASK_ID)"
      echo ""
      echo "Penalty Options:"
      echo "  --fillup_eot_penalty PENALTY EOT token penalty for fillup phase (default: $FILLUP_EOT_PENALTY)"
      echo "  --denoise_eot_penalty PENALTY EOT token penalty for denoise phase (default: $DENOISE_EOT_PENALTY)"
      echo "  --eot_penalty PENALTY        Set both EOT penalties to same value (for compatibility)"
      echo ""
      echo "Denoising Options:"
      echo "  --denoise_max_iter ITER     Maximum denoising iterations (default: $DENOISE_MAX_ITER)"
      echo "  --start_remask_ratio RATIO  Starting remask ratio for annealing (default: $START_REMASK_RATIO)"
      echo "  --end_remask_ratio RATIO    Ending remask ratio for annealing (default: $END_REMASK_RATIO)"
      echo "  --decay_schedule SCHEDULE  Annealing schedule: 'cosine' or 'exponential' (default: $DECAY_SCHEDULE)"
      echo "  --denoise_update_rate RATE Update rate for denoising (default: $DENOISE_UPDATE_RATE)"
      echo "  --denoise_temp TEMP         Denoising sampling temperature (default: $DENOISE_TEMP)"
      echo "  --repeat_converge_threshold THR  Convergence threshold for position locking (default: $REPEAT_CONVERGE_THRESHOLD)"
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
      echo ""
      echo "Examples:"
      echo "  # Basic evaluation with denoising"
      echo "  $0 --tasks gsm8k_cot --limit 100"
      echo ""
      echo "  # Custom denoising parameters"
      echo "  $0 --tasks gsm8k_cot --denoise_max_iter 20 --start_remask_ratio 0.9 --end_remask_ratio 0.2"
      echo ""
      echo "  # Disable convergence (set to very high value)"
      echo "  $0 --repeat_converge_threshold 999"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create output filename with all relevant parameters
OUTPUT_FILENAME="${TASKS}_llada_denoise_steps_${STEPS}_tokens_${MAX_NEW_TOKENS}_block_${BLOCK_LENGTH}_temp_${TEMPERATURE}_fillup_eot_${FILLUP_EOT_PENALTY}_denoise_eot_${DENOISE_EOT_PENALTY}_iter_${DENOISE_MAX_ITER}_converge_${REPEAT_CONVERGE_THRESHOLD}_seed_${SEED}"

echo "==== Running LLaDA-Tolerator evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$MAX_NEW_TOKENS, block_length=$BLOCK_LENGTH, temperature=$TEMPERATURE, denoise_temp=$DENOISE_TEMP"
echo "Tolerator: fillup_eot=$FILLUP_EOT_PENALTY, denoise_eot=$DENOISE_EOT_PENALTY, max_iter=$DENOISE_MAX_ITER, ratio=$START_REMASK_RATIO->$END_REMASK_RATIO, converge_threshold=$REPEAT_CONVERGE_THRESHOLD"
echo "Seed: $SEED"
echo "===================================="

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model llada_denoise \
  --model_args "pretrained=$MODEL_NAME,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,gen_length=$MAX_NEW_TOKENS,block_length=$BLOCK_LENGTH,dtype=bfloat16,temperature=$TEMPERATURE,cfg_scale=$CFG_SCALE,remasking=$REMASKING,mask_id=$MASK_ID,fillup_eot_penalty=$FILLUP_EOT_PENALTY,denoise_eot_penalty=$DENOISE_EOT_PENALTY,denoise_max_iter=$DENOISE_MAX_ITER,start_remask_ratio=$START_REMASK_RATIO,end_remask_ratio=$END_REMASK_RATIO,decay_schedule=$DECAY_SCHEDULE,denoise_update_rate=$DENOISE_UPDATE_RATE,denoise_temp=$DENOISE_TEMP,repeat_converge_threshold=$REPEAT_CONVERGE_THRESHOLD" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEWSHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${OUTPUT_FILENAME}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template