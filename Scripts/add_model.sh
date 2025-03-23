#!/bin/bash
# Script to run all necessary tests for a new model and update comparison plots
print_header() {
    echo -e "\n$(printf '=%.0s' {1..80})"
    echo "  $1"
    echo "$(printf '=%.0s' {1..80})"
}

run_command() {
    local cmd="$1"
    local description="$2"
    
    if [ -n "$description" ]; then
        print_header "$description"
    fi
    
    echo -e "\nRunning: $cmd"
    start_time=$(date +%s)
    
    eval $cmd
    exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Command completed in $duration seconds"
    
    if [ $exit_code -ne 0 ]; then
        echo "Command failed with exit code $exit_code"
        return 1
    fi
    
    return 0
}

create_directories() {
    for dir in "../Generations" "../Surprisals" "../UID_Analysis" "../UID_Comparison"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

directory_exists() {
    if [ -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

MODEL=""
GENERATE=false
NUM_EXAMPLES=300
TEMP=0.7
TOP_P=0.95
SYSTEM_PROMPT="Provided only the following article title and first sentence, complete the rest of the article from this moment onwards:"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [options]"
    echo "Options:"
    echo "  --generate              Generate new outputs instead of just analyzing"
    echo "  --num-examples <num>    Number of examples to generate/analyze (default: 300)"
    echo "  --temp <value>          Temperature for generation (default: 0.7)"
    echo "  --top-p <value>         Top-p for generation (default: 0.95)"
    echo "  --system-prompt <text>  System prompt for generation"
    exit 1
fi

MODEL="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --generate)
            GENERATE=true
            ;;
        --num-examples)
            NUM_EXAMPLES="$2"
            shift
            ;;
        --temp)
            TEMP="$2"
            shift
            ;;
        --top-p)
            TOP_P="$2"
            shift
            ;;
        --system-prompt)
            SYSTEM_PROMPT="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

create_directories

if [ "$GENERATE" = true ]; then
    CMD="uv run prompter.py $MODEL -g $NUM_EXAMPLES -t $TEMP -p $TOP_P -s \"$SYSTEM_PROMPT\" -v"
    run_command "$CMD" "Generating and analyzing $NUM_EXAMPLES examples for $MODEL"
else
    CMD="uv run prompter.py $MODEL --analyze-only -g $NUM_EXAMPLES -v"
    run_command "$CMD" "Analyzing existing outputs for $MODEL"
fi

if [ $? -ne 0 ]; then
    echo "Failed to complete surprisal analysis. Aborting."
    exit 1
fi

CMD="uv run analyze_uid.py --input-dir \"../Surprisals/$MODEL\" --output-dir \"../UID_Analysis/$MODEL\""
run_command "$CMD" "Analyzing UID metrics for $MODEL"

if [ $? -ne 0 ]; then
    echo "Failed to complete UID analysis. Aborting."
    exit 1
fi

DEFAULT_MODELS=("human_texts" "llama-2-7b-32k-instruct" "llama-2-7b@q8_0" "mistral-7b-v0.1")

if [ "$MODEL" = "mistralai_-_mistral-7b-instruct-v0.1" ]; then
    NEW_DEFAULT_MODELS=()
    for m in "${DEFAULT_MODELS[@]}"; do
        if [ "$m" != "mistral-7b-instruct-v0.3" ]; then
            NEW_DEFAULT_MODELS+=("$m")
        fi
    done
    if [[ ! " ${NEW_DEFAULT_MODELS[@]} " =~ " $MODEL " ]]; then
        NEW_DEFAULT_MODELS+=("$MODEL")
    fi
    DEFAULT_MODELS=("${NEW_DEFAULT_MODELS[@]}")
else
    if [[ ! " ${DEFAULT_MODELS[@]} " =~ " $MODEL " ]]; then
        DEFAULT_MODELS+=("$MODEL")
    fi
fi

EXISTING_MODELS=()
for m in "${DEFAULT_MODELS[@]}"; do
    if directory_exists "../UID_Analysis/$m"; then
        EXISTING_MODELS+=("$m")
    else
        echo "Warning: No UID analysis found for $m"
    fi
done

DIRECTORIES=""
for m in "${EXISTING_MODELS[@]}"; do
    DIRECTORIES="$DIRECTORIES \"../UID_Analysis/$m\""
done

if [ ${#EXISTING_MODELS[@]} -ge 2 ]; then
    CMD="uv run compare_uid.py --directories $DIRECTORIES --output-dir \"../UID_Comparison/all_models\""
    run_command "$CMD" "Comparing all models"
fi

if [[ " ${EXISTING_MODELS[@]} " =~ " human_texts " ]]; then
    CMD="uv run compare_uid.py --directories $DIRECTORIES --output-dir \"../UID_Comparison/human_vs_all\""
    run_command "$CMD" "Comparing human texts vs all models"
    
    for m in "${EXISTING_MODELS[@]}"; do
        if [ "$m" != "human_texts" ]; then
            CMD="uv run compare_uid.py --directories \"../UID_Analysis/human_texts\" \"../UID_Analysis/$m\" --output-dir \"../UID_Comparison/human_vs_$m\""
            run_command "$CMD" "Comparing human texts vs $m"
        fi
    done
fi

LLAMA_MODELS=()
MISTRAL_MODELS=()

for m in "${EXISTING_MODELS[@]}"; do
    if [[ "$m" == *"llama"* ]]; then
        LLAMA_MODELS+=("$m")
    fi
    if [[ "$m" == *"mistral"* ]]; then
        MISTRAL_MODELS+=("$m")
    fi
done

if [ ${#LLAMA_MODELS[@]} -ge 2 ]; then
    DIRECTORIES=""
    for m in "${LLAMA_MODELS[@]}"; do
        DIRECTORIES="$DIRECTORIES \"../UID_Analysis/$m\""
    done
    CMD="uv run compare_uid.py --directories $DIRECTORIES --output-dir \"../UID_Comparison/llama_models\""
    run_command "$CMD" "Comparing Llama models"
fi

if [ ${#MISTRAL_MODELS[@]} -ge 2 ]; then
    DIRECTORIES=""
    for m in "${MISTRAL_MODELS[@]}"; do
        DIRECTORIES="$DIRECTORIES \"../UID_Analysis/$m\""
    done
    CMD="uv run compare_uid.py --directories $DIRECTORIES --output-dir \"../UID_Comparison/mistral_models\""
    run_command "$CMD" "Comparing Mistral models"
fi

echo -e "\nAll tests and comparisons completed successfully!"
echo -e "\nSummary of comparisons generated:"
echo "- All models comparison"

if [[ " ${EXISTING_MODELS[@]} " =~ " human_texts " ]]; then
    echo "- Human texts vs all models"
    for m in "${EXISTING_MODELS[@]}"; do
        if [ "$m" != "human_texts" ]; then
            echo "- Human texts vs $m"
        fi
    done
fi

if [ ${#LLAMA_MODELS[@]} -ge 2 ]; then
    echo "- Llama models comparison"
fi

if [ ${#MISTRAL_MODELS[@]} -ge 2 ]; then
    echo "- Mistral models comparison"
fi

exit 0