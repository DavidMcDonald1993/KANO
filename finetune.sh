# exp_id=tox21
# exp_id=sider
# exp_id=bbbp
exp_id=chembl_28_single_protein_homo_sapiens

# max_data_size required for current memory limits 

python train.py \
    --data_path ./data/${exp_id}.csv \
    --dataset_type classification \
    --max_data_size 350000 \
    --metric prc-auc \
    --epochs 1 \
    --num_runs 1 \
    --gpu 0 \
    --batch_size 128 \
    --seed 51 \
    --init_lr 1e-4  \
    --split_type 'scaffold_balanced' \
    --step functional_prompt \
    --exp_name finetune \
    --exp_id ${exp_id} \
    --checkpoint_path "./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"