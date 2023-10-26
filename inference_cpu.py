
import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd

from chemprop.data.utils import get_data

from chemprop.utils import load_checkpoint
from chemprop.train.predict import predict
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    date_of_model_training = "1018"
    model_id = "chembl_28_single_protein_homo_sapiens"

    save_dir = f"dumped/{date_of_model_training}-finetune/{model_id}/run_0/model_0"
    cuda = False
    logger = None 
    prompt = False #?
    batch_size = 128
    scaler = None # required for regression
    use_compound_names = True


    # read in dataset and convert to chemprop.data.data.MoleculeDataset
    data_filename = args.input
    print ("Loading SMILES from", data_filename)
    test_data = get_data(
        path=data_filename,
        skip_invalid_smiles=True,
        use_compound_names=use_compound_names, # does the file also include compound names?
    )

    # load model
    model_filename = os.path.join(save_dir, "model.pt")
    print ("Loading model from", model_filename)

    model, model_args = load_checkpoint(model_filename, cuda=cuda, logger=logger, return_args=True,)

    test_preds = predict(
        model=model,
        prompt=prompt,
        data=test_data,
        batch_size=batch_size,
        scaler=scaler
    )

    test_preds = pd.DataFrame(test_preds, index=test_data.compound_names(), columns=model_args.task_names)

    output_filename = args.output
    print ("Writing predictions of shape", test_preds.shape, "to", output_filename)
    test_preds.to_csv(output_filename)