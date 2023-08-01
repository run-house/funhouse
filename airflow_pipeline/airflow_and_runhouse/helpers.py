import pickle
import torch
import runhouse as rh
from transformers import AutoModelForSequenceClassification, get_scheduler


def tokenize_dataset():
    """Tokenize the dataset using the BERT tokenizer, and save resulting dataset to disk on the cluster."""
    # data = ....
    return rh.table(data="data").write().save()


def fine_tune_model():
    """Fine tune the model on the preprocessed dataset."""
    # model = ...
    # Save as anonymous blob to local file system on the cluster
    return rh.blob(data=pickle.dumps("model")).write().save()


def get_model(num_labels, model_id='bert-base-cased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return model


def get_optimizer(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer
