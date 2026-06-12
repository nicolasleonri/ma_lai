#!/bin/bash

RESULTS_DIR="./results/csv/multi_label"
OUTPUT_CSV="./results/csv/multi_label/aggregated_results.csv"

# Remove existing CSV if it exists
[ -f "$OUTPUT_CSV" ] && rm "$OUTPUT_CSV"

# Header row
echo "prefix,prefix2,augmentation,aa,ht,cv,up,samples,metric,date,eval_accuracy,eval_mean_task_accuracy,eval_macro_f1" > "$OUTPUT_CSV"

# Loop through all JSON files
for filepath in "$RESULTS_DIR"/*.json; do
    filename=$(basename "$filepath")
    basename_no_ext="${filename%.json}"

    # Parse fixed fields from the right
    date="${basename_no_ext##*_}"
    basename_no_date="${basename_no_ext%_*}"

    metric="${basename_no_date##*_}"
    basename_no_metric="${basename_no_date%_*}"

    IFS='_' read -ra parts <<< "$basename_no_date"
    len=${#parts[@]}

    # Check for 'macro_f1'
    if [[ "${parts[$((len-2))]}" == "macro" && "${parts[$((len-1))]}" == "f1" ]]; then
        metric="macro_f1"
        basename_no_metric="${basename_no_date%_macro_f1}"
    else
        metric="${parts[$((len-1))]}"
        basename_no_metric="${basename_no_date%_*}"
    fi

    samples="${basename_no_metric##*_}"
    basename_no_samples="${basename_no_metric%_*}"

    up="${basename_no_samples##*_}"
    basename_no_up="${basename_no_samples%_*}"

    cv="${basename_no_up##*_}"
    basename_no_cv="${basename_no_up%_*}"

    ht="${basename_no_cv##*_}"
    basename_no_ht="${basename_no_cv%_*}"

    aa="${basename_no_ht##*_}"
    basename_no_aa="${basename_no_ht%_*}"

    # Remaining part is prefix + augmentation
    prefix_and_aug="${basename_no_aa}"

    # Split first two as prefix/prefix2, rest as augmentation
    prefix=$(echo "$prefix_and_aug" | cut -d'_' -f1)
    prefix2=$(echo "$prefix_and_aug" | cut -d'_' -f2)
    augmentation=$(echo "$prefix_and_aug" | cut -d'_' -f3-)
    [ -z "$augmentation" ] && augmentation="None"

    # Extract only the desired JSON values
    eval_accuracy=$(grep -oP '"eval_accuracy":\s*\K[0-9.]+|null' "$filepath")
    eval_mean_task_accuracy=$(grep -oP '"eval_mean_task_accuracy":\s*\K[0-9.]+|null' "$filepath")
    eval_macro_f1=$(grep -oP '"eval_macro_f1":\s*\K[0-9.]+|null' "$filepath")

    # Append row to CSV
    echo "$prefix,$prefix2,$augmentation,$aa,$ht,$cv,$up,$samples,$metric,$date,$eval_accuracy,$eval_mean_task_accuracy,$eval_macro_f1" >> "$OUTPUT_CSV"
done

echo "✅ Aggregated JSON files into $OUTPUT_CSV"
