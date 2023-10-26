dump_path="dumped/single_protein_targets"
split_type="scaffold_balanced"

# pretrained CMPN path
checkpoint_path="./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"

# process alphabetically in order to know which model to remove if cancelling
targets=$(ls data/single_protein_targets/P*.csv | head -n 10000)

for target in $targets;
do
    
    # basename 
    target="${target##*/}"
    # remove extension
    exp_id="${target%.*}"

    # check for model
    if [ -f ${dump_path}/*-finetune/${exp_id}/run_0/model_0/model.pt ];
    then 
        echo $exp_id exists
        continue
    fi

    data_path="./data/single_protein_targets/${exp_id}.csv"

    python train.py \
        --data_path ${data_path} \
        --dataset_type classification \
        --split_sizes 0.8 0.1 0.1 \
        --max_data_size 1500000 \
        --metric prc-auc \
        --epochs 100 \
        --num_runs 1 \
        --gpu 0 \
        --batch_size 128 \
        --seed 51 \
        --init_lr 1e-4  \
        --split_type ${split_type} \
        --step functional_prompt \
        --exp_name finetune \
        --exp_id ${exp_id} \
        --dump_path ${dump_path} \
        --checkpoint_path ${checkpoint_path} 

done
