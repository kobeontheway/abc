OUTPUT=result

hub autofinetune autofinetune.py \
    --param_file=hparam.yaml \
    --gpu=0 \
    --popsize=15 \
    --round=30 \
    --output_dir=${OUTPUT} \
    --evaluator=fulltrail \
    --tuning_strategy=pshe2
