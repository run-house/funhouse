"""
The below is pseudo code showing how we can integrate Runhouse with Airflow. We define our DAG where
each node is running Runhouse functions, or microservices. In this particular example we are taking a simple
two-step workflow consisting of data preprocessing and model fine-tuning.

By using Airflow to orchestrate Runhouse code, we save the extra translation step required by
Airflow (and other orchestration tools), to break up existing code into the "glue code" required for each
task. By not having to translate to the code into a DSL, we keep the code in its original form,
ensuring re-usability and saving us this extra translation step while still getting the benefits of
the orchestration tool (scheduling, triggering, monitoring, etc.).
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datasets import load_dataset
import runhouse as rh

from airflow.airflow_and_runhouse.helpers import tokenize_dataset, fine_tune_model, get_optimizer, get_model

dag = DAG("prediction_pipeline",
          description="ML Prediction Pipeline",
          # predict every day
          schedule_interval="30 15 * * *",
          catchup=True
          )


def preprocess():
    cpu = rh.cluster("^rh-32-cpu").up_if_not()
    preproc = rh.function(fn=tokenize_dataset,
                          env=['datasets', 'transformers'],
                          name="BERT_preproc_32cpu").to(cpu).save()

    # Not being saved, just a helper here to load the dataset on the cluster instead of doing it locally
    remote_load_dataset = rh.function(fn=load_dataset, dryrun=True).to(preproc.system)

    # Calls the function async, leaves the result on the cluster, and gives us back a Run object, which
    # we can then pass to other functions on the cluster, and they'll auto-resolve to our object via a Ray ObjectRef
    yelp_train_run = remote_load_dataset.run("yelp_review_full", split='train[:10%]')
    yelp_test_run = remote_load_dataset.run("yelp_review_full", split='test[:10%]')

    # converts the table's file references to remote file references without bouncing the data back to our laptop
    preprocessed_yelp_train = preproc(yelp_train_run.name)
    preprocessed_yelp_test = preproc(yelp_test_run.name)

    preprocessed_yelp_train.write().save(name="preprocessed-yelp-train")
    preprocessed_yelp_test.write().save(name="preprocessed-yelp-test")


def training():
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x',
                                                                             instance_type='A10:1').up_if_not()

    # Load the preprocessed table we built in the preprocess task - we'll later stream the data directly on the cluster
    preprocessed_yelp = rh.table(name="preprocessed-yelp-train")

    ft_model = rh.function(fn=fine_tune_model, load_secrets=True,name='finetune_ddp_1gpu').to(gpu).save()

    # Send get_model and get_optimizer to the cluster so we can call .run() and instantiate them on the cluster
    model_on_gpu = rh.function(fn=get_model).to(gpu)
    optimizer_on_gpu = rh.function(fn=get_optimizer).to(gpu)

    # Receive an object ref for the model and optimizer
    bert_model_run = model_on_gpu.run(num_labels=5, model_id='bert-base-cased')
    adam_optimizer_run = optimizer_on_gpu.run(model=bert_model_run.name, lr=5e-5)

    trained_model = ft_model(bert_model_run.name,
                             adam_optimizer_run.name,
                             preprocessed_yelp,
                             num_epochs=3,
                             batch_size=32)

    # Copy model from the cluster to s3 bucket, and save the model's metadata to Runhouse RNS for re-loading later
    trained_model.to('s3').save(name='yelp_fine_tuned_bert')


with dag:
    # Instantiating a PythonOperator class results in the creation of a
    # task object, which ultimately becomes a node in DAG objects.
    preprocess = PythonOperator(task_id="preprocess",
                                python_callable=preprocess,
                                )

    fine_tune = PythonOperator(task_id="fine_tune",
                               python_callable=training,
                               )

    preprocess >> fine_tune
