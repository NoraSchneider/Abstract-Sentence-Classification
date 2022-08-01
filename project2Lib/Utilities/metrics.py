from datasets import load_metric

# Helper function for computing the weighted F1-score
def compute_f1(eval_pred):
    predictions, labels = eval_pred
    pred = predictions.argmax(axis=-1)
    return load_metric("f1").compute(predictions=pred, references=labels, average="weighted")

# Helper function for computing the accuracy
def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    pred = predictions.argmax(axis=-1)
    return load_metric("accuracy").compute(predictions=pred, references=labels)
