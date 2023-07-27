import runhouse as rh
from transformers import AutoModelForSequenceClassification, get_scheduler
import ray.cloudpickle as pickle
import torch
from tqdm.auto import tqdm


# Based on https://huggingface.co/docs/transformers/training#train-in-native-pytorch

def fine_tune_model(model, optimizer, preprocessed_table, num_epochs=3, batch_size=8):
    # Set data format to pytorch tensors
    preprocessed_table.stream_format = 'torch'
    device = torch.device("cuda")
    model.to(device)

    num_training_steps = num_epochs * len(preprocessed_table)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    print("Training model.")
    model.train()

    # https://huggingface.co/course/chapter8/2?fw=pt
    for epoch in range(num_epochs):
        for batch in preprocessed_table.stream(batch_size=batch_size, as_dict=True):
            batch = {k: v.to(device).long() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(batch_size)

    # Save as anonymous blob to local file system on the cluster ( in '.cache/blobs/..')
    return rh.blob(data=pickle.dumps(model)).write()


def get_model(num_labels, model_id='bert-base-cased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return model


def get_optimizer(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') \
        else rh.cluster(name='rh-a10x', instance_type='A100:1').save()

    # Load the preprocessed table object based on p01 - we'll stream the data directly on the cluster later on
    preprocessed_yelp = rh.table(name="preprocessed-yelp-train")

    ft_model = rh.function(fn=fine_tune_model, load_secrets=True, name='finetune_ddp_1gpu').to(gpu).save()

    # The load_secrets argument above will load the secrets onto the cluster from your Runhouse account (api.run.house),
    # and will only work if you've already uploaded secrets to runhouse (e.g. during `runhouse login`). You need your
    # SkyPilot ssh keys on the gpu cluster because we're streaming in the table directly from the 32-cpu cluster.

    # If you'd like to run this tutorial without an account or saved secrets, you can uncomment this line:
    ft_model.send_secrets()

    # Send get_model and get_optimizer to the cluster. Save them for future re-use.
    model_on_gpu = rh.function(get_model, name="get_model").to(gpu).save()
    optimizer_on_gpu = rh.function(get_optimizer, name="model_optimizer").to(gpu).save()

    # Create Run objects which also generate an object ref on the cluster for the model and optimizer
    bert_model_run = model_on_gpu.run(run_name="bert_model", num_labels=5, model_id='bert-base-cased')
    adam_optimizer_run = optimizer_on_gpu.run(run_name="adam_optimizer", model=bert_model_run.name, lr=5e-5)

    trained_model = ft_model(bert_model_run.name,
                             adam_optimizer_run.name,
                             preprocessed_yelp,
                             num_epochs=3,
                             batch_size=32)

    # Copy model from the cluster to s3 bucket, and save the model's metadata to Runhouse RNS for re-loading later
    trained_model.to('s3').save(name='yelp_fine_tuned_bert')
