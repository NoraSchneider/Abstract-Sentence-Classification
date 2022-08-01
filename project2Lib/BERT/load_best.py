import json
import pandas as pd
from pathlib import Path
from tensorflow.python.summary.summary_iterator import summary_iterator
from typing import List

hyper_params =    [
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "lr_scheduler_type",
    "warmup_ratio",
    "warmup_steps",
    "metric_for_best_model",
    "greater_is_better",
    "optim",
    "adafactor",
    "num_train_epochs",
    "weight_decay",
]

def load_best_bert(model_dir : Path, return_args : List[str] = hyper_params):

    top_args = {}
    top_score = -float("inf")
    for p in model_dir.joinpath("runs").glob("**/*/events.out.tfevents*"):
        
        # There might be problems with some files due to crashes
        # Only consider well-working files
        try:

            # Get the top score of the current run
            score = max(v.simple_value for e in summary_iterator(str(p)) \
                                      for v in e.summary.value \
                                      if v.tag == "eval/f1")
            
            # Compute the corresponding hyper parameters for that run
            # Assert that only one set of hyper parameters are saved
            args = [json.loads(v.tensor.string_val[0].decode("utf-8")) \
                                      for e in summary_iterator(str(p)) \
                                      for v in e.summary.value \
                                      if v.tag == "args/text_summary"]
            assert len(args) == 1
            args = args[0]

            # Update best arguments
            if score > top_score:
                top_score = score
                top_args = args

        # Ignore bad files
        except:
            continue

    # Return both top score and top arguments
    # Only return relevant keys
    return top_score, {key: top_args[key] for key in return_args}


def load_bert_runs(model_dir : Path, return_args : List[str] = hyper_params):

    runs = []
    for p in model_dir.joinpath("runs").glob("**/*/events.out.tfevents*"):
        
        # There might be problems with some files due to crashes
        # Only consider well-working files
        try:

            # Get the top score of the current run
            score = max(v.simple_value for e in summary_iterator(str(p)) \
                                      for v in e.summary.value \
                                      if v.tag == "eval/f1")
            
            # Compute the corresponding hyper parameters for that run
            # Assert that only one set of hyper parameters are saved
            args = [json.loads(v.tensor.string_val[0].decode("utf-8")) \
                                      for e in summary_iterator(str(p)) \
                                      for v in e.summary.value \
                                      if v.tag == "args/text_summary"]
            assert len(args) == 1

            # Add run with relevant fields
            run = {key: args[0][key] for key in return_args}
            run["score"] = score
            runs.append(run)


        # Ignore bad files
        except:
            continue

    # Return DataFrame of Runs
    return pd.DataFrame(runs)
