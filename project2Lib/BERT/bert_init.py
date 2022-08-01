from transformers import AutoModelForSequenceClassification

def bert_init(model_checkpoint : str, freeze_bert : bool, num_labels : int):

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False

    return model
