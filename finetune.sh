exp_name="properties"
# exp_name="finetune"

exp_id=tox21
# exp_id=sider
# exp_id=bbbp

# exp_id=chembl_28_single_protein_homo_sapiens
# exp_id=chembl_28_single_protein_homo_sapiens_pidgin
# exp_id=chembl_28_single_protein_homo_sapiens_pidgin_no_missing

data_path="data/${exp_id}.csv"

split_type="scaffold_balanced"

# pretrained 
checkpoint_path="./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"

# finetuned
# finetune_path="dumped/1022-${exp_name}/${exp_id}/run_0/model_0/model.pt"

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
    --exp_name ${exp_name} \
    --exp_id ${exp_id} \
    --transpose_evaluation_matrix \
    --checkpoint_path ${checkpoint_path} #\
    # --finetune_path ${finetune_path}