#!/bin/sh
DATASET=$1
NUM=5
if [ ! $DATASET ]; then
    echo "Usage ./bin/prune.sh  [ldc93s1 | ted] "
    exit
fi;
EXTRA_PARAM="${@:2}"
if [ $DATASET == "ldc93s1" ]; then
    DATA_DIR='./data/ldc93s1'
    TRAIN_FILE='data/ldc93s1/ldc93s1.csv'
    DEV_FILE='data/ldc93s1/ldc93s1.csv'
    TEST_FILE='data/ldc93s1/ldc93s1.csv'
    IMPORTER='bin/import_ldc93s1.py'
    EPOCH=50
    TRAIN_BATCH_SIZE=1
    DEV_BATCH_SIZE=1
    TEST_BATCH_SIZE=1

fi;

set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/ldc93s1/ldc93s1.csv" ]; then
    #echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u $IMPORTER $DATA_DIR
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c "from xdg import BaseDirectory as xdg; print(xdg.save_data_path(\"deepspeech/$DATASET\"))")
fi

train()
{
    echo "Test only $1"
    TEST_ONLY=$1
    if [ $TEST_ONLY == true ]; then
        python -u DeepSpeech.py \
            --train_files $TRAIN_FILE \
            --dev_files $DEV_FILE \
            --test_files $TEST_FILE \
            --train_batch_size $TRAIN_BATCH_SIZE \
            --dev_batch_size $DEV_BATCH_SIZE \
            --test_batch_size $TEST_BATCH_SIZE \
            --n_hidden 494 \
            --epoch $EPOCH \
            --checkpoint_dir "$checkpoint_dir" \
            --weight_sharing True \
            --train False \
            "$EXTRA_PARAM"
    else

        python -u DeepSpeech.py \
            --train_files $TRAIN_FILE \
            --dev_files $DEV_FILE \
            --test_files $TEST_FILE \
            --train_batch_size $TRAIN_BATCH_SIZE \
            --dev_batch_size $DEV_BATCH_SIZE \
            --test_batch_size $TEST_BATCH_SIZE \
            --n_hidden 494 \
            --epoch $EPOCH \
            --checkpoint_dir "$checkpoint_dir" \
            --weight_sharing True \
            "$EXTRA_PARAM"
    fi;

}

prune()
{
    python -u prune.py \
        --checkpoint_dir "$checkpoint_dir" \
        --save_checkpoint True
}

infer()
{
    python -u inference.py \
        --dataset $DATASET \
        "$EXTRA_PARAM"
}

main()
{
    count=0
    train false
    while [ $count -le $NUM ]
    do
        prune
        if [ $count -eq $NUM ]; then
            echo "Last iteration. Just test only"
            train true
        else
            train false
        fi;
        #prune
        infer
        count=`expr $count + 1`
    done
}

main
