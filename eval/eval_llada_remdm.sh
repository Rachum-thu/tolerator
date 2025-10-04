#!/usr/bin/env bash

# LLaDA-ReMDM Evaluation Script
# This script runs lm-eval with LLaDA model enhanced with ReMDM sampling strategies

export HF_ALLOW_CODE_EVAL="1"  # Required for code evaluation tasks like mbpp

# Set default parameters
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
STRATEGY="remdm-loop"
TASKS="humaneval_instruct"
NUM_FEW_SHOT=0
BATCH_SIZE=1
STEPS=32
GEN_LENGTH=256  # max_new_tokens
BLOCK_LENGTH=256
TEMPERATURE=0.3
CFG_SCALE=0.0
REMASKING="low_confidence"  # Used for original strategy comparison
MASK_ID=126336
ETA=0.02
NUCLEUS_P=0.9
T_ON=0.6
T_OFF=0.4
ALPHA_ON=0.95
LIMIT=32  # for quick test
SEED=42
PORT=12345
DEVICE="cuda"
OUTPUT_DIR="../output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --num_fewshot)
            NUM_FEW_SHOT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --gen_length)
            GEN_LENGTH="$2"
            shift 2
            ;;
        --block_length)
            BLOCK_LENGTH="$2"
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
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --cfg_scale)
            CFG_SCALE="$2"
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
        --remasking)
            REMASKING="$2"
            shift 2
            ;;
        --mask_id)
            MASK_ID="$2"
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
        --limit)
            LIMIT="$2"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL_NAME         HuggingFace model path (default: $MODEL_NAME)"
            echo "  --strategy STRATEGY         ReMDM strategy: remdm-cap, remdm-rescale, remdm-conf, remdm-loop, original (default: $STRATEGY)"
            echo "  --tasks TASKS              Comma-separated tasks to evaluate (default: $TASKS)"
            echo "  --num_fewshot N            Number of few-shot examples (default: $NUM_FEW_SHOT)"
            echo "  --batch_size N             Batch size (default: $BATCH_SIZE)"
            echo "  --steps N                  Number of sampling steps (default: $STEPS)"
            echo "  --gen_length N             Generation length (default: $GEN_LENGTH)"
            echo "  --block_length N           Block length for semi-AR generation (default: $BLOCK_LENGTH)"
            echo "  --eta FLOAT                ReMDM eta parameter (default: $ETA)"
            echo "  --nucleus_p FLOAT          Top-p sampling threshold (default: $NUCLEUS_P)"
            echo "  --temperature FLOAT        Sampling temperature (default: $TEMPERATURE)"
            echo "  --cfg_scale FLOAT          Classifier-free guidance scale (default: $CFG_SCALE)"
            echo "  --t_on FLOAT               Turn-on time for remdm-loop (default: $T_ON)"
            echo "  --t_off FLOAT              Turn-off time for remdm-loop (default: $T_OFF)"
            echo "  --alpha_on FLOAT           Alpha parameter for remdm-loop (default: $ALPHA_ON)"
            echo "  --limit N                  Limit number of examples to evaluate (default: $LIMIT)"
            echo "  --output_dir DIR           Output directory for results (default: $OUTPUT_DIR)"
            echo "  --seed SEED                Random seed for reproducibility (default: $SEED)"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic evaluation with limit"
            echo "  $0 --strategy remdm-cap --tasks gsm8k_cot --limit 100"
            echo "  # Custom parameters"
            echo "  $0 --strategy remdm-cap --tasks hellaswag,arc_easy --eta 0.01 --steps 256 --limit 500"
            echo "  # Quick test (very small)"
            echo "  $0 --strategy remdm-cap --limit 5 --steps 16 --gen_length 64"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==== Running LLaDA-ReMDM evaluation ===="
echo "Generation: steps=$STEPS, max_new_tokens=$GEN_LENGTH, block_length=$BLOCK_LENGTH, temperature=$TEMPERATURE"
echo "ReMDM: strategy=$STRATEGY, eta=$ETA, nucleus_p=$NUCLEUS_P, t_on=$T_ON, t_off=$T_OFF, alpha_on=$ALPHA_ON"
echo "Seed: $SEED"
echo "===================================="

# Construct model arguments (following vanilla LLaDA style)
MODEL_ARGS="pretrained=$MODEL_NAME,trust_remote_code=True"
MODEL_ARGS="$MODEL_ARGS,max_new_tokens=$GEN_LENGTH,steps=$STEPS,gen_length=$GEN_LENGTH"
MODEL_ARGS="$MODEL_ARGS,block_length=$BLOCK_LENGTH,dtype=bfloat16"
MODEL_ARGS="$MODEL_ARGS,temperature=$TEMPERATURE,cfg_scale=$CFG_SCALE"
MODEL_ARGS="$MODEL_ARGS,remasking_strategy=$STRATEGY,mask_id=$MASK_ID"
MODEL_ARGS="$MODEL_ARGS,eta=$ETA,nucleus_p=$NUCLEUS_P"
MODEL_ARGS="$MODEL_ARGS,t_on=$T_ON,t_off=$T_OFF,alpha_on=$ALPHA_ON"

PYTHONPATH=. accelerate launch --main_process_port $PORT -m lm_eval \
  --model llada_remdm \
  --model_args "$MODEL_ARGS" \
  --tasks $TASKS \
  --limit $LIMIT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_fewshot $NUM_FEW_SHOT \
  --seed $SEED \
  --output_path "${OUTPUT_DIR}/${TASKS}_llada_remdm_steps_${STEPS}_tokens_${GEN_LENGTH}_block_${BLOCK_LENGTH}_temp_${TEMPERATURE}_strategy_${STRATEGY}_eta_${ETA}_seed_${SEED}" \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template