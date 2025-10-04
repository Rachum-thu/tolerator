#!/usr/bin/env bash

# Dream-ReMDM Evaluation Script
# This script runs lm-eval with Dream model enhanced with ReMDM sampling strategies

export HF_ALLOW_CODE_EVAL="1"  # Required for code evaluation tasks like mbpp

# Default values - Dream generation parameters (from eval_dream_denoise.sh)
MODEL_NAME="Tolerator-org/Dream-v0-Instruct-7B-ReMDM"
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

# ReMDM specific parameters (from batch_eval_llada_remdm.sh and our implementation)
REMDM_STRATEGY="remdm-loop"  # remdm-loop or original
ETA=0.02
NUCLEUS_P=0.9
T_ON=0.6
T_OFF=0.4
ALPHA_ON=0.95

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
    --remdm_strategy|--strategy)
      REMDM_STRATEGY="$2"
      shift 2
      ;;
    --eta)
      ETA="$2"
      shift 2
      ;;
    --nucleus_p)
      NUCLEUS_P="$2"
      shift 2
      ;;
    --t_on)
      T_ON="$2"
      shift 2
      ;;
    --t_off)
      T_OFF="$2"
      shift 2
      ;;
    --alpha_on)
      ALPHA_ON="$2"
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
      echo "Dream-ReMDM Evaluation Script"
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
      echo "ReMDM Options:"
      echo "  --remdm_strategy|--strategy STRATEGY  ReMDM strategy: original or remdm-loop (default: $REMDM_STRATEGY)"
      echo "  --eta ETA                   ReMDM eta parameter (default: $ETA)"
      echo "  --nucleus_p NUCLEUS_P       Nucleus sampling threshold (default: $NUCLEUS_P)"
      echo "  --t_on T_ON                 Turn-on time for remdm-loop (default: $T_ON)"
      echo "  --t_off T_OFF               Turn-off time for remdm-loop (default: $T_OFF)"
      echo "  --alpha_on ALPHA_ON         Alpha parameter for remdm-loop (default: $ALPHA_ON)"
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
      echo "  # Basic evaluation"
      echo "  $0"
      echo "  # Custom ReMDM parameters"
      echo "  $0 --eta 0.05 --t_on 0.8 --t_off 0.2 --steps 64"
      echo "  # Different algorithm with ReMDM"
      echo "  $0 --alg maskgit_plus --remdm_strategy remdm-loop"
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

echo "==== Running Dream-ReMDM evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$MAX_NEW_TOKENS, temperature=$TEMPERATURE, alg=$ALG"
echo "ReMDM: strategy=$REMDM_STRATEGY, eta=$ETA, nucleus_p=$NUCLEUS_P, t_on=$T_ON, t_off=$T_OFF, alpha_on=$ALPHA_ON"
echo "Penalties: repetition=$REPETITION_PENALTY, fillup_eot=$FILLUP_EOT_PENALTY"
echo "Seed: $SEED"
echo "===================================="

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model dream_remdm \
  --model_args "pretrained=$MODEL_NAME,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,diffusion_steps=${STEPS},dtype=bfloat16,temperature=$TEMPERATURE,top_p=$TOP_P,alg=$ALG,repetition_penalty=$REPETITION_PENALTY,fillup_eot_penalty=$FILLUP_EOT_PENALTY,remdm_strategy=$REMDM_STRATEGY,eta=$ETA,nucleus_p=$NUCLEUS_P,t_on=$T_ON,t_off=$T_OFF,alpha_on=$ALPHA_ON" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEWSHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${TASKS}_dream_remdm_steps_${STEPS}_tokens_${MAX_NEW_TOKENS}_temp_${TEMPERATURE}_strategy_${REMDM_STRATEGY}_eta_${ETA}_seed_${SEED}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template