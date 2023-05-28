ROOT_DIR="/mnt/d/PhD"

# SIZE="20"
declare -a INDEXES=("0" "1" "2")
declare -a SIZES=("20" "50" "100" "200")
# declare -a INDEXES=("1")


for SIZE in "${SIZES[@]}"
do
    for INDEX in "${INDEXES[@]}"
    do
        EXP_ID="exp_${SIZE}_${INDEX}"
        echo "==================================================="
        echo "$EXP_ID"
        echo "==================================================="
        python aum-st.py \
            --augmentation_dir="${ROOT_DIR}/AUM-ST-Augmentations" \
            --input_dir="${ROOT_DIR}/AUM-ST-Dataset" \
            --train_file="labeled_${SIZE}_${INDEX}.csv" \
            --validation_path="${ROOT_DIR}/AUM-ST-Dataset/valid.csv" \
            --test_path="${ROOT_DIR}/AUM-ST-Dataset/test.csv" \
            --num_classes=3 \
            --experiment_id="${EXP_ID}" \
            --aum_save_dir="${ROOT_DIR}/AUM-ST-Results/AUM-ST-Results-tibi-20apr/${EXP_ID}" \
            --intermediate_model_path="${ROOT_DIR}/AUM-ST-Results/AUM-ST-Results-tibi-26apr/${EXP_ID}/" \
            --tensorboard_dir="${ROOT_DIR}/AUM-ST-Results/AUM-ST-Results-tibi-20apr/tboard" \
            --action="eval_full" \
            --weak_augmentation_max_strength=0 \
            --strong_augmentation_min_strength=0 \
            --strong_augmentation_max_strength=0
    done
done