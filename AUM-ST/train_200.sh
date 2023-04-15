ROOT_DIR="/mnt/d/PhD"

SIZE="200"
declare -a INDEXES=("0" "1" "2")

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
        --aum_save_dir="${ROOT_DIR}/AUM-ST-Results/${EXP_ID}" \
        --intermediate_model_path="${ROOT_DIR}/AUM-ST-Results/${EXP_ID}" \
        --tensorboard_dir="${ROOT_DIR}/AUM-ST-Results/tboard"
done
