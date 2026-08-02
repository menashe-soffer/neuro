#!/bin/bash

# Define the variable arrays
SESSIONS=(1)
#(1 2)
#EPOCH_OPTIONS=("all" "odd" "even")
#EPOCH_OPTIONS=("all" "first" "second")
EPOCH_OPTIONS=("odd" "even")
HR_SEL_OPTIONS=("max" "short")

echo "Starting batch run of 18 script combinations..."
echo "=============================================="

# Counter for tracking progress
RUN_COUNT=0

# Loop through NUM_SESSIONS (2 options)
for hr_sel in "${HR_SEL_OPTIONS[@]}"; do
    for sess in "${SESSIONS[@]}"; do
        # Loop through SELECT_BY_EPOCHS (3 options)
        for select_ep in "${EPOCH_OPTIONS[@]}"; do
            # Loop through CALC_BY_EPOCHS (3 options)
            for calc_ep in "${EPOCH_OPTIONS[@]}"; do

                # Skip if select_ep and calc_ep are the same
                if [[ "$select_ep" != "$calc_ep" ]]|| true; then
                    
                    ((RUN_COUNT++))
                    echo "----------------------------------------"
                    echo "Job $RUN_COUNT/18: Sessions=$sess | Select=$select_ep | Calc=$calc_ep | HRSEL=$hr_sel"
                    echo "----------------------------------------"
                    
                    # Invoke your script with the current combination parameters
                    python generate_psth_from_all.py \
                        --NUM_SESSIONS "$sess" \
                        --SELECT_BY_EPOCHS "$select_ep" \
                        --CALC_BY_EPOCHS "$calc_ep" \
                        --HR_SELECT_TYPE "$hr_sel"
                
                fi
                    
            done
        done
    done
    src_folder="/home/labs/malach/sofferme/figs/${sess} sessions"
    tgt_folder="/home/labs/malach/sofferme/figs/${sess} sessions_${hr_sel}"
    if [ -d "$tgt_folder" ]; then
        rm -rf "$tgt_folder"
    fi
    mv "$src_folder" "$tgt_folder"
done

echo "=============================================="
echo "All 18 invocations completed successfully."

