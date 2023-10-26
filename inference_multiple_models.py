import os, glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd

from chemprop.data.utils import get_data

from chemprop.utils import load_checkpoint, load_scalers
from chemprop.train.predict import predict

if __name__ == "__main__":

    # cuda = True
    cuda = False

    logger = None 
    prompt = False #?
    batch_size = 128
    use_compound_names = True

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
    
    # concatenate predictions of multiple models
    model_filenames = glob.glob(f"dumped/single_protein_targets/*-finetune/*/run_0/model_0/model.pt")

    # concatenate using axus=1
    all_predictions = []
   
    for model_filename in model_filenames:

        # if "P10275" not in model_filename: # AR
        #     continue

        # load model
        print ("Loading model from", model_filename)
        model, model_args = load_checkpoint(model_filename, cuda=cuda, logger=logger, return_args=True,)
        scaler, features_scaler = load_scalers(model_filename)

        if features_scaler is not None:
            # reloading is required
            test_data = get_data(
                path=data_filename,
                skip_invalid_smiles=True,
                use_compound_names=use_compound_names, # does the file also include compound names?
                logger=logger,
            )
            test_data.normalize_features(features_scaler)

        test_preds = predict(
            model=model,
            prompt=prompt,
            data=test_data,
            batch_size=batch_size,
            scaler=scaler,
        )

        test_preds = pd.DataFrame(
            test_preds, 
            index=test_data.compound_names(), 
            columns=model_args.task_names)

        all_predictions.append(test_preds)

    # concatenate
    all_predictions = pd.concat(all_predictions, axis=1)

    output_dir = os.path.join(
        "predictions",
        "single_protein_targets",
        )
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, data_filename + ".gz")
    print ("Writing predictions of shape", all_predictions.shape, "to", output_filename)
    all_predictions.to_csv(output_filename)