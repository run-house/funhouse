import boto3
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import runhouse as rh
import os


def download_data(data_set_path, split, data_set_name=None):
    """
    :param data_set_path: the name / path of the hugging face data set.
    :param split: which part of the data set to download: train, validation or test.
    :param data_set_name: the name of the dataset's subset to be downloaded (if such exists).
    :return: a loaded hugging face dataset.
    """
    dataset = load_dataset(data_set_path, data_set_name, split=split)
    return dataset


def preprocess(hf_token_parameter_name):
    """
    :param hf_token_parameter_name: the name of the parameter name in SSM which stores your hugging face token.
    :return: Dataframe, which includes preprocessed 'openbookqa' data.
    """
    ssm_client = boto3.client('ssm')
    huggingface_token = ssm_client.get_parameter(Name=hf_token_parameter_name, WithDecryption=True)
    login(token=huggingface_token['Parameter']['Value'])

    dataset = download_data("openbookqa", "train", "additional")

    dataset_df = pd.DataFrame([x for x in dataset])
    dataset_df = dataset_df[dataset_df["fact1"].notna()]
    dataset_df = dataset_df[dataset_df['humanScore'] > 0.81]
    dataset_df = dataset_df[dataset_df['clarity'] > 1.5]
    return dataset_df


if __name__ == '__main__':
    # name of the parameters in ssm which stores your hugging face token
    hf_token_name = "/huggingFace/sasha/token"
    preprocess_lambda = rh.aws_lambda_fn(fn=preprocess,
                                         name="funhouse_preprocess_test",
                                         env=['datasets', 'huggingface_hub', 'pandas']).save()
    preprocessed_data = preprocess_lambda(hf_token_name)
    print(preprocessed_data.head())
    print(preprocessed_data.shape)
