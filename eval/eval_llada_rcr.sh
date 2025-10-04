#!/usr/bin/env bash

export HF_ALLOW_CODE_EVAL="1"
# Default values
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
TASKS="humaneval_instruct"
# Generation parameters
STEPS=32
MAX_NEW_TOKENS=256
BLOCK_LENGTH=256
TEMPERATURE=0.3
CFG_SCALE=0.0
REMASKING="low_confidence"  # Keep for compatibility
MASK_ID=126336
# NEW RCR-specific parameters
RCR=false
CONF_ALG="llada"
MODE="linear"
TOP_P=0.9
TOP_K=""
# Evaluation parameters
LIMIT=32  # for quick test
BATCH_SIZE=1
NUM_FEWSHOT=0
PORT=12334
DEVICE="cuda"
SEED=42
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
    # NEW RCR-specific options
    --rcr)
      RCR="$2"
      shift 2
      ;;
    --conf_alg)
      CONF_ALG="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "LLaDA-RCR Evaluation Script"
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
      echo "  --remasking STRATEGY        [Legacy] Remasking strategy: 'low_confidence' or 'random' (default: $REMASKING)"
      echo "  --mask_id ID                Mask token ID (default: $MASK_ID)"
      echo ""
      echo "RCR-specific Options:"
      echo "  --rcr BOOL                  Enable Running Confidence Remasking (default: $RCR)"
      echo "  --conf_alg ALG              Confidence algorithm: 'llada', 'entropy', 'topk_margin', 'random' (default: $CONF_ALG)"
      echo "  --mode MODE                 Scheduling mode: 'linear', 'cosine', 'pow2', etc. (default: $MODE)"
      echo "  --top_p TOP_P               Top-p sampling threshold (default: $TOP_P)"
      echo "  --top_k TOP_K               Top-k sampling threshold (default: $TOP_K)"
      echo ""
      echo "Evaluation Options:"
      echo "  --limit LIMIT              Limit number of examples (default: $LIMIT)"
      echo "  --batch_size SIZE          Batch size (default: $BATCH_SIZE)"
      echo "  --num_fewshot NUM          Number of few-shot examples (default: $NUM_FEWSHOT)"
      echo "  --port PORT                Main process port (default: $PORT)"
      echo "  --device DEVICE            Device to use (default: $DEVICE)"
      echo "  --seed SEED                Random seed (default: $SEED)"
      echo "  --output_dir DIR           Output directory (default: $OUTPUT_DIR)"
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

echo "==== Running LLaDA-RCR evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$MAX_NEW_TOKENS, block_length=$BLOCK_LENGTH, temperature=$TEMPERATURE"
echo "RCR: enabled=$RCR, conf_alg=$CONF_ALG, mode=$MODE, top_p=$TOP_P, top_k=$TOP_K"
echo "Seed: $SEED"
echo "===================================="

# Build model args string
MODEL_ARGS="pretrained=$MODEL_NAME,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,gen_length=$MAX_NEW_TOKENS,block_length=$BLOCK_LENGTH,dtype=bfloat16,temperature=$TEMPERATURE,cfg_scale=$CFG_SCALE,remasking=$REMASKING,mask_id=$MASK_ID,rcr=$RCR,conf_alg=$CONF_ALG,mode=$MODE"

# Add optional parameters if provided
if [[ -n "$TOP_P" ]]; then
    MODEL_ARGS="$MODEL_ARGS,top_p=$TOP_P"
fi

if [[ -n "$TOP_K" ]]; then
    MODEL_ARGS="$MODEL_ARGS,top_k=$TOP_K"
fi

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model llada_rcr \
  --model_args "$MODEL_ARGS" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEWSHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${TASKS}_llada_rcr_steps_${STEPS}_tokens_${MAX_NEW_TOKENS}_block_${BLOCK_LENGTH}_temp_${TEMPERATURE}_rcr_${RCR}_conf_${CONF_ALG}_mode_${MODE}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template