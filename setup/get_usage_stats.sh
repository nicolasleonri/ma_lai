#!/usr/bin/env bash

# Usage: ./extract_metrics.sh <output.csv> <jobid1> <jobid2> ...
# Example: ./extract_metrics.sh report.csv 24123090 24123094

OUTFILE="$1"
shift

# Fields to extract (comma-separated for sacct)
FIELDS="JobID,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,CPUTime,ConsumedEnergy,ConsumedEnergyRaw,MaxDiskRead,MaxDiskWrite"

# Write header with semicolons
echo "$FIELDS" | tr ',' ';' > "$OUTFILE"

# Extract job data and convert "|" to ";" in the output
for JOB in "$@"; do
    sacct -j "$JOB" --parsable2 --noheader --format="$FIELDS" \
        | tr '|' ';' >> "$OUTFILE"
done
