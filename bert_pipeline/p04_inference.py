import pickle

import runhouse as rh
import torch

from p01_preprocess import tokenizer


def predict_sentiment(samples):
    model = rh.blob(name='yelp_fine_tuned_bert')
    sa_model = rh.get_pinned_object(model.name)
    if sa_model is None:
        sa_model = pickle.loads(model.data).to("cuda")
        rh.pin_to_memory(model.name, sa_model)

    inputs = tokenizer(samples, padding="max_length", truncation=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        logits = sa_model(**inputs).logits

    predicted_class_id = logits.argmax(dim=1).tolist()
    return predicted_class_id


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') \
        else rh.cluster(name='rh-a10x', instance_type='A100:1').save()

    bert_sa_service = rh.function(fn=predict_sentiment).to(system=gpu, reqs=['s3fs'])
    new_examples = [
        'This place is excellent!',
        'The service was horrible.',
        'The appetizers were hit or miss.',
        'The elote ribs are a must.',
    ]

    sentiment_scores = bert_sa_service(new_examples)
    print('Test samples and sentiment scores:')
    [print(f'{sample}: {score}') for (sample, score) in zip(new_examples, sentiment_scores)]

    bert_sa_service.save(name='ft_pipeline_sa_service')
