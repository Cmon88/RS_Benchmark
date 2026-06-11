#!/bin/bash
# Run sample_dense.py for a manual list of datasets

CONFIG="${1:-test.yaml}"  # default config, override with: ./run_all_datasets_sample.sh test_TO.yaml
shift 1                   # remaining args forwarded to sample_dense.py (e.g. --output-dir ../dataset_sampled_100k)
EXTRA_ARGS="$@"

datasets=(
    # "Amazon_Books"
    # "Amazon_Digital_Music"
    # "Amazon_Grocery_and_Gourmet_Food"
    # "Amazon_Health_and_Personal_Care"
    # "Amazon_Movies_and_TV"
    # "anime"
    # "BeerAdvocate"
    # "Behance"
    # "book-crossing"
    # "DianPing"
    # "douban"
    # "epinions"
    # "Food"
    # "GoodReads"
    # "jester"
    "KDD2010-algebra2006_2007"
    # "lastfm"
    # "ml-1m"
    # "ModCloth"
    # "netflix"
    # "pinterest"
    # "RateBeer"
    # "RentTheRunway"
    # "steam"
    # "Twitch-100k"
    # "yahoo-music"
    # "yelp2022"
)

total=${#datasets[@]}
echo "Found $total datasets"
echo "Config: $CONFIG"
echo "========================================"

failed=()

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    echo ""
    echo "[$((i+1))/$total] Sampling dataset: $dataset"
    echo "----------------------------------------"

    python sample_dense.py --dataset "$dataset" --config "$CONFIG" $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "  ERROR: sample_dense.py failed for dataset '$dataset'"
        failed+=("$dataset")
    else
        echo "  Finished dataset: $dataset"
    fi
done

echo ""
echo "========================================"
echo "All datasets processed ($total total)."
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed datasets (${#failed[@]}): ${failed[*]}"
else
    echo "No failures."
fi
