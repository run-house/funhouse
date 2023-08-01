import os
import runhouse as rh
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)


def tokenize_dataset(hf_dataset):
    tokenized_datasets = hf_dataset.map(tokenize_function,
                                        input_columns=['text'],
                                        num_proc=os.cpu_count(),
                                        batched=True)

    # https://github.com/huggingface/transformers/issues/12631
    # Remove the text column because the model does not accept raw text as an input
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Rename the label column to labels because the model expects the argument to be named labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # We'll return the table object here so the user of this service can save it to whatever datastore they
    # prefer, under whichever Runhouse name they prefer. We need to call write() to write it down to the local
    # filesystem on the cluster, as we're only returning a reference to the user rather than the full dataset.
    preprocessed_data = rh.table(data=tokenized_datasets).write().save()
    print(f"Saved preprocessed data on cluster to path: {preprocessed_data.path}")
    return preprocessed_data


if __name__ == "__main__":
    cpu = rh.cluster("^rh-32-cpu").up_if_not().save()

    preproc = rh.function(tokenize_dataset, name="BERT_preproc_32cpu").to(cpu, env=['datasets', 'transformers']).save()

    # Helper to load a dataset on a cluster instead of locally
    load_dataset = rh.function(load_dataset, name="load_dataset").to(preproc.system).save()

    # Notice how we call this function with `.run()` - this calls the function async, leaves the result on the
    # cluster, and gives us back a Run object. We can then pass in the name of this Run
    # to the other functions on the cluster, which will auto-resolve to our object.
    yelp_train_run = load_dataset.run("yelp_review_full", split='train[:10%]')
    yelp_test_run = load_dataset.run("yelp_review_full", split='test[:10%]')

    # converts the table's file references to sftp file references without copying it
    preprocessed_yelp_train = preproc(yelp_train_run.name)
    preprocessed_yelp_test = preproc(yelp_train_run.name)

    preprocessed_yelp_test.stream_format = 'torch'
    batches = preprocessed_yelp_test.stream(batch_size=32)
    for batch in batches:
        print(batch)
        break

    # Write the train and data to the local filesystem on the cluster & save config to Den
    preprocessed_yelp_train.write().save(name="preprocessed-yelp-train")
    preprocessed_yelp_test.write().save(name="preprocessed-yelp-test")
