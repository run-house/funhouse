import json
import boto3
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import runhouse as rh


def download_data(data_set_path, split, data_set_name=None):
    """
    :param data_set_path: the name / path of the hugging face data set.
    :param split: which part of the data set to download: train, validation or test.
    :param data_set_name: the name of the dataset's subset to be downloaded (if such exists).
    :return: a loaded hugging face dataset.
    """
    dataset = load_dataset(data_set_path, data_set_name, split=split)
    return dataset


def preprocess(hf_token):
    """
    :param hf_token: a token for logining into hugging face. 
    :return: Dataframe, which includes preprocessed 'openbookqa' data.
    """

    login(hf_token)

    dataset = download_data("openbookqa", "train", "additional")

    dataset_df = pd.DataFrame([x for x in dataset])
    dataset_df = dataset_df[dataset_df["fact1"].notna()]
    dataset_df = dataset_df[dataset_df['humanScore'] > 0.81]
    dataset_df = dataset_df[dataset_df['clarity'] > 1.5]
    return dataset_df.to_json(orient="records")


if __name__ == '__main__':
    # name of the parameters in ssm which stores your hugging face token
    huggingface_token = "add_your_hugging_face_token_here"
    preprocess_lambda = rh.aws_lambda_fn(fn=preprocess,
                                         name="funhouse-preprocess-data",
                                         env=['pandas', 'datasets', 'huggingface_hub']).save()
    data = preprocess_lambda(huggingface_token)
    data = json.loads(data)
    df = pd.DataFrame(data)
    print(df.head())


