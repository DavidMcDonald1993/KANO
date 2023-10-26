import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd

from chemprop.data.utils import get_data

from chemprop.utils import load_checkpoint, load_scalers
from chemprop.train.predict import predict

if __name__ == "__main__":

    date_of_model_training = "1025"

    model_id = "chembl_28_single_protein_homo_sapiens" # same as NPAI
    # model_id = "chembl_28_single_protein_homo_sapiens_pidgin" # same as PIDGIN4 (~1.2 million molecules, sparse labels)
    # model_id = "chembl_28_single_protein_homo_sapiens_pidgin_no_missing" # same as PIDGIN4 (~1.2 million molecules, dense labels)

    save_dir = f"dumped/{date_of_model_training}-finetune/{model_id}/run_0/model_0"
    cuda = True
    # cuda = False
    logger = None 
    prompt = False #?
    batch_size = 128
    use_compound_names = True

    # load model
    model_filename = os.path.join(save_dir, "model.pt")
    print ("Loading model from", model_filename)

    model, model_args = load_checkpoint(model_filename, cuda=cuda, logger=logger, return_args=True,)
    scaler, features_scaler = load_scalers(model_filename)

    # read in dataset and convert to chemprop.data.data.MoleculeDataset
    # data_filename = "not_in_chembl_28_homo_sapiens.csv"
    data_filename = "AR.csv"

    print ("Loading SMILES from", data_filename)
    test_data = get_data(
        path=data_filename,
        skip_invalid_smiles=True,
        use_compound_names=use_compound_names, # does the file also include compound names?
        logger=logger,
    )

    if features_scaler is not None:
        test_data.normalize_features(features_scaler)

    test_preds = predict(
        model=model,
        prompt=prompt,
        data=test_data,
        batch_size=batch_size,
        scaler=scaler,
    )

    test_preds = pd.DataFrame(test_preds, index=test_data.compound_names(), columns=model_args.task_names)

    output_dir = os.path.join(
        "predictions",
        date_of_model_training,
        model_id,
        )
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, data_filename + ".gz")
    print ("Writing predictions of shape", test_preds.shape, "to", output_filename)
    test_preds.to_csv(output_filename)