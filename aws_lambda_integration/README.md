# 👩🏻‍💻 Runhouse & AWS Lambdas

Runhouse is designed to make interacting with shared, heterogeneous
hardware and data resources feel easy and natural. That includes supporting
serverless computing capabilities, such as AWS Lambda functions. Runhouse aims to
make it easier to create, deploy and call serverless functions. 

This tutorial will provide examples of how to create and use AWS Lambdas with 
the help of [Runhouse](https://www.run.house/docs/api/python/function#runhouse.aws_lambda_fn). The examples are using [Hugging Face](https://huggingface.co/) dataset and pretrained
model.

## Using a Hugging Face 🤗 dataset for simple preprocess function

In this example we are sending a callable function to AWS Lambda, invoking it, and
displaying the results. During runtime, the function will use some python 
libraries such as pandas, and therefore they are passed as an argument to
the aws_lambda constructor. 

The function loads the [openbookqa](https://huggingface.co/datasets/openbookqa) dataset from hugging face, 
preprocess it, and returns the preprocessed data as a pandas Data Frame. 

## Loading and using a Hugging Face 🤗 model

In this example, we are loading from Hugging Face a [BERT model](https://huggingface.co/sshleifer/distilbart-cnn-12-6) 
for summarizing a text, and apply it on a given textual intput.
<br/>
<br/>
### How to run those examples?
Good news - the given examples are ready to be used! The only thing you need to do is to set them 
locally on your machine:

* Clone the funhouse project to your working directory:
```
git clone git@github.com:run-house/funhouse.git
```
* Go to aws_lambda_integration folder in the funhouse project:
```
cd /path/to/your/working/dir/funhouse/aws_lambda_integration
```
* make sure the relevant libraries are installed in your environment:
```
pip install -r requirements.txt
```
* Execute any example you like:
```
python preproccess_function.py <MY-HUGGING-FACE-TOKEN>

python summarization_hf_to_lambda.py
```
Please note that these examples are two stand-alone examples, and are not depending on one another. Meaning, there is 
no particular order in which the examples should be run.

The examples can be run from any IDE or python environment. 





