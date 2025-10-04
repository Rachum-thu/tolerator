#!/usr/bin/env bash

# Dream-RCR Evaluation Script
# This script runs lm-eval with Dream model enhanced with RCR (Running Confidence Remasking)

export HF_ALLOW_CODE_EVAL="1"  # Required for code evaluation tasks like mbpp

# Default values - Dream generation parameters
MODEL_NAME="Tolerator-org/Dream-v0-Instruct-7B-RCR"
TASKS="humaneval_instruct"
# Generation parameters
STEPS=32
MAX_NEW_TOKENS=256
TEMPERATURE=0.3
TOP_P=0.9
ALG="entropy"
# Penalty parameters (keeping from Dream)
REPETITION_PENALTY=1.0
FILLUP_EOT_PENALTY=1.0

# RCR specific parameters (aligned with LLaDA-RCR defaults)
RCR=true  # Enable RCR
CONF_ALG="llada"  # Confidence algorithm: llada, entropy, topk_margin, random
MODE="linear"  # Future parameter: linear, cosine, pow2, sqrt

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
      echo "Dream-RCR Evaluation Script"
      echo ""
      echo "Model Options:"
      echo "  --model MODEL_NAME          Model to evaluate (default: $MODEL_NAME)"
      echo "  --tasks TASKS               Tasks to evaluate (default: $TASKS)"
      echo ""
      echo "Dream Generation Options:"
      echo "  --steps|--diffusion_steps STEPS  Diffusion steps (default: $STEPS)"
      echo "  --max_new_tokens|--max_tokens TOKENS  Max new tokens (default: $MAX_NEW_TOKENS)"
      echo "  --temperature TEMP          Temperature for sampling (default: $TEMPERATURE)"
      echo "  --top_p TOP_P              Top-p for sampling (default: $TOP_P)"
      echo "  --alg|--algorithm ALG       Algorithm to use: origin,maskgit_plus,topk_margin,entropy (default: $ALG)"
      echo ""
      echo "Penalty Options:"
      echo "  --repetition_penalty|--penalty PENALTY  Repetition penalty (default: $REPETITION_PENALTY)"
      echo "  --fillup_eot_penalty PENALTY    EOT token penalty for fillup phase (default: $FILLUP_EOT_PENALTY)"
      echo ""
      echo "RCR Options:"
      echo "  --rcr RCR                   Enable/disable RCR: true or false (default: $RCR)"
      echo "  --conf_alg CONF_ALG         Confidence algorithm: llada,entropy,topk_margin,random (default: $CONF_ALG)"
      echo "  --mode MODE                 Mode parameter: linear,cosine,pow2,sqrt (default: $MODE)"
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
      echo "  # Basic RCR evaluation with default llada confidence"
      echo "  $0"
      echo "  # RCR with entropy confidence algorithm"
      echo "  $0 --conf_alg entropy"
      echo "  # RCR with topk_margin confidence"
      echo "  $0 --conf_alg topk_margin --alg maskgit_plus"
      echo "  # Disable RCR (vanilla Dream)"
      echo "  $0 --rcr false"
      echo "  # Quick test with limit"
      echo "  $0 --limit 100 --steps 16"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "==== Running Dream-RCR evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$MAX_NEW_TOKENS, temperature=$TEMPERATURE, alg=$ALG"
echo "RCR: enabled=$RCR, conf_alg=$CONF_ALG, mode=$MODE"
echo "Penalties: repetition=$REPETITION_PENALTY, fillup_eot=$FILLUP_EOT_PENALTY"
echo "Seed: $SEED"
echo "===================================="

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model dream_rcr \
  --model_args "pretrained=$MODEL_NAME,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,diffusion_steps=${STEPS},dtype=bfloat16,temperature=$TEMPERATURE,top_p=$TOP_P,alg=$ALG,repetition_penalty=$REPETITION_PENALTY,fillup_eot_penalty=$FILLUP_EOT_PENALTY,rcr=$RCR,conf_alg=$CONF_ALG,mode=$MODE" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEWSHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${TASKS}_dream_rcr_steps_${STEPS}_tokens_${MAX_NEW_TOKENS}_temp_${TEMPERATURE}_rcr_${RCR}_conf_${CONF_ALG}_mode_${MODE}_seed_${SEED}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template