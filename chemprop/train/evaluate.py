import logging
from typing import Callable, List

import torch.nn as nn

from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler

from functools import partial
from concurrent.futures import ProcessPoolExecutor

def evaluate_one_task(
    valid_targets,
    valid_preds,
    dataset_type: str,
    metric_func: Callable,
    info,
    ):

     # Skip if all targets or preds are identical, otherwise we'll crash during classification
    if dataset_type == 'classification':
        nan = False
        if all(target == 0 for target in valid_targets) or all(target == 1 for target in valid_targets):
            nan = True
            info('Warning: Found a task with targets all 0s or all 1s')
        if all(pred == 0 for pred in valid_preds) or all(pred == 1 for pred in valid_preds):
            nan = True
            info('Warning: Found a task with predictions all 0s or all 1s')

        if nan:
            return float('nan')

    if len(valid_targets) == 0:
        return None # filter out None values

    if dataset_type == 'multiclass':
        return metric_func(valid_targets, valid_preds, labels=list(range(len(valid_preds[0]))))
    else:
        return metric_func(valid_targets, valid_preds)


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         dataset_type: str,
                         logger: logging.Logger = None,
                         transpose_evaluation_matrix: bool = False,
                         ) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    info = logger.info if logger is not None else print

    # TODO: per-molecule evalution (rather than per-task)

    if transpose_evaluation_matrix: # transpose molecules and tasks to evaluate per-molecule rather than per-task
        info("Transposing evaluation matrix")
        import pandas as pd
        preds = pd.DataFrame(preds).T.values.tolist()
        targets = pd.DataFrame(targets).T.values.astype(float).tolist()

        num_tasks = len(preds[0])

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # multiprocessing evaluation 
    n_proc = 5
    info(f"Evaluating using {n_proc} process(es)")
    
    with ProcessPoolExecutor(max_workers=n_proc) as p:

        results = p.map(
            partial(
                evaluate_one_task,
                dataset_type=dataset_type,
                metric_func=metric_func,
                info=info,
            ),
            valid_targets,
            valid_preds,
        )

        # filter out None
        results = list(filter(lambda e: e is not None, results))

    # # Compute metric
    # results = []

    # for i in range(num_tasks):
    #     # # Skip if all targets or preds are identical, otherwise we'll crash during classification
    #     if dataset_type == 'classification':
    #         nan = False
    #         if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
    #             nan = True
    #             # info('Warning: Found a task with targets all 0s or all 1s')
    #         if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
    #             nan = True
    #             # info('Warning: Found a task with predictions all 0s or all 1s')

    #         if nan:
    #             results.append(float('nan'))
    #             continue

    #     if len(valid_targets[i]) == 0:
    #         continue

    #     if dataset_type == 'multiclass':
    #         results.append(metric_func(valid_targets[i], valid_preds[i], labels=list(range(len(valid_preds[i][0])))))
    #     else:
    #         results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             prompt: bool,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,
             transpose_evaluation_matrix: bool = False,
             ) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds = predict(
        model=model,
        prompt=prompt,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets()

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger,
        transpose_evaluation_matrix=transpose_evaluation_matrix,
    )

    return results
