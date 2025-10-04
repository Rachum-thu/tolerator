#!/bin/bash

export HF_ALLOW_CODE_EVAL="1"  # Required for code evaluation tasks like mbpp

# Default values
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
TASKS="humaneval_instruct"  # other options including mbpp_instruct, gsm8k_cot, gpqa_main_cot_zeroshot, triviaqa
# Generation parameters
STEPS=32
MAX_NEW_TOKENS=256
BLOCK_LENGTH=256
TEMPERATURE=0.3
CFG_SCALE=0.0
REMASKING="low_confidence"
MASK_ID=126336
# Evaluation parameters
LIMIT=32  # for quick test
BATCH_SIZE=1
NUM_FEWSHOT=0
SEED=42
PORT=12346
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
      echo "LLaDA Evaluation Script"
      echo ""
      echo "Model Options:"
      echo "  --model MODEL_NAME          Model to evaluate (default: $MODEL_NAME)"
      echo "  --tasks TASKS               Tasks to evaluate (default: $TASKS)"
      echo ""
      echo "Generation Options:"
      echo "  --steps STEPS               LLaDA sampling steps (default: $STEPS)"
      echo "  --max_new_tokens|--gen_length LENGTH  Generated answer length (default: $MAX_NEW_TOKENS)"
      echo "  --block_length LENGTH       Block length for semi-autoregressive (default: $BLOCK_LENGTH)"
      echo "  --temperature TEMP          Categorical distribution sampling temperature (default: $TEMPERATURE)"
      echo "  --cfg_scale SCALE           Classifier-free guidance scale (default: $CFG_SCALE)"
      echo "  --remasking STRATEGY        Remasking strategy: 'low_confidence' or 'random' (default: $REMASKING)"
      echo "  --mask_id ID                Mask token ID (default: $MASK_ID)"
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

echo "==== Running LLaDA evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$MAX_NEW_TOKENS, block_length=$BLOCK_LENGTH, temperature=$TEMPERATURE"
echo "Seed: $SEED"
echo "===================================="

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model llada \
  --model_args "pretrained=$MODEL_NAME,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,gen_length=$MAX_NEW_TOKENS,block_length=$BLOCK_LENGTH,dtype=bfloat16,temperature=$TEMPERATURE,cfg_scale=$CFG_SCALE,remasking=$REMASKING,mask_id=$MASK_ID" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEWSHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${TASKS}_llada_steps_${STEPS}_tokens_${MAX_NEW_TOKENS}_block_${BLOCK_LENGTH}_temp_${TEMPERATURE}_seed_${SEED}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template