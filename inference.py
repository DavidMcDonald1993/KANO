
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd

from chemprop.data.utils import get_data

from chemprop.utils import load_checkpoint
from chemprop.train.predict import predict
    
if __name__ == "__main__":

    save_dir = "dumped/1018-finetune/sider/run_0/model_0"
    cuda = True
    logger = None 
    prompt = False #?
    batch_size = 128
    scaler = None # required for regression
    use_compound_names = True


    # read in dataset and convert to chemprop.data.data.MoleculeDataset
    data_filename = "test_input_head.csv"
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

    # import numpy as np
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])

    # raise Exception(params)

    test_preds = predict(
        model=model,
        prompt=prompt,
        data=test_data,
        batch_size=batch_size,
        scaler=scaler
    )

    test_preds = pd.DataFrame(test_preds, index=test_data.compound_names(), columns=model_args.task_names)

    output_filename = "output.csv"
    print ("Writing predictions to", output_filename)
    test_preds.to_csv(output_filename)