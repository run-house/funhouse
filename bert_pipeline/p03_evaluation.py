import runhouse as rh
from datasets import load_metric
import torch
import ray.cloudpickle as pickle
from tqdm.auto import tqdm


def evaluate_model(model, preprocessed_test_set, batch_size=32):
    model = pickle.loads(model.data)
    preprocessed_test_set.stream_format = 'torch'
    device = torch.device("cuda")
    model.to(device)

    metric = load_metric("accuracy")
    progress_bar = tqdm(range(len(preprocessed_test_set)))
    print("Evaluating model.")
    model.eval()

    for batch in preprocessed_test_set.stream(batch_size=batch_size, as_dict=True):
        batch = {k: v.to(device).long() for k, v in batch.items()}
        labels = batch.pop("labels")

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        progress_bar.update(batch_size)

    accuracy = metric.compute()
    return accuracy


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') \
        else rh.cluster(name='rh-a10x', instance_type='A100:1').save()

    # Note: Depending on the python version used may need to change the default version of boto3 installed
    # https://stackoverflow.com/questions/75887656/botocore-package-in-lambda-python-3-9-runtime-return-error-cannot-import-name
    model_eval = rh.function(evaluate_model, name='evaluate_model').to(gpu, env=['s3fs', 'scikit-learn',
                                                                                 'boto3==1.26.90'])

    # Load model we created in P02 (note: we'll unpickle the file on the cluster later)
    trained_model = rh.blob(name='yelp_fine_tuned_bert')
    preprocessed_yelp_test = rh.table(name="preprocessed-yelp-test")

    test_accuracy = model_eval(trained_model, preprocessed_yelp_test, batch_size=64)
    print('Test accuracy:', test_accuracy)

    model_eval.save("bert_ft_eval")
