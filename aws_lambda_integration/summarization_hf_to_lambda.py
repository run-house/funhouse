import runhouse as rh
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
from pathlib import Path


def summarize_txt(article):

    # sshleifer/distilbart-cnn-12-6 is a text summarize model presented in Hugging Face.

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    max_token_length = 512  # set by the model

    # Tokenize the input text with the specified max_length
    tokens = tokenizer(article, max_length=max_token_length, return_tensors='pt', truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(tokens['input_ids'])

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


if __name__ == '__main__':
    reqs = ['transformers',
            'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117']
    summarize_txt_lambda = rh.aws_lambda_fn(fn=summarize_txt,
                                            env=reqs,
                                            name="summarize_txt_hf").save()
    with open(str(Path.cwd() / "text_to_sum.txt")) as text2sum:
        txt_to_sum = text2sum.read()
    summery = summarize_txt_lambda(txt_to_sum)
    summery = json.loads(summery)['body']
    print(f"Article's summery is: {summery}")
