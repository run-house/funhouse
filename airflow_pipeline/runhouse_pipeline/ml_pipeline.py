import pickle

import runhouse as rh

from airflow_vs_runhouse.runhouse_pipeline.helpers import preprocess_raw_data, split_data, fit_and_save_model, \
    predict_test_wt_arima, measure_accuracy, load_raw_data


# Based on Delivery Hero Airflow ML Workshop, specifically the training_pipeline.py DAG
# https://github.com/deliveryhero/pyconde2019-airflow-ml-workshop/blob/master/dags/training_pipeline.py


def preprocessing_and_data_split(raw_df, cpu):
    # Send the function for loading the dataset to the cluster along with the requirements it needs to run
    preprocessed_data_on_cpu = rh.function(preprocess_raw_data, name="preprocess_data").to(cpu, env=["pmdarima"]).save()

    # Run the preprocessing on the cluster, which returns a remote reference to the dataset saved on the cluster
    dataset_ref_on_cpu = preprocessed_data_on_cpu(raw_df)
    print(f"Saved dataset on cluster to path: {dataset_ref_on_cpu.path}")

    # Run the data splitting on the cluster, which returns a remote reference to the train + test data on the cluster
    split_data_on_cpu = rh.function(split_data, name="split_data").to(cpu).save()
    train_data_ref, test_data_ref = split_data_on_cpu(preprocessed_dataset_ref=dataset_ref_on_cpu)

    print(f"Saved train data on the cluster to path: {train_data_ref.path}")
    print(f"Saved test data on the cluster to path: {test_data_ref.path}")

    return train_data_ref, test_data_ref


def model_training(gpu, train_data, test_data):
    train_model_on_gpu = rh.function(fn=fit_and_save_model, name="fit_and_save_model").to(gpu, env=["pmdarima"]).save()

    # Send the SkyPilot ssh keys to the gpu cluster because we're streaming in the train / test data directly
    # from the 32-cpu cluster
    train_model_on_gpu.send_secrets()

    # Run the training on the cluster
    model = train_model_on_gpu(train_dataset_ref=train_data)
    print(f"Saved model on cluster to path: {model.path}")

    predict_on_gpu = rh.function(predict_test_wt_arima, name="ml_pipeline_predict").to(gpu, env=["pmdarima"]).save()
    test_predictions = predict_on_gpu(test_dataset_ref=test_data)
    print(f"Saved test data predictions on cluster to path: {test_predictions.path}")

    return model, test_predictions


def run_pipeline():
    """
    The Runhouse pipeline consists of the same steps outlined in the Airflow DAG:
    preprocess >> split data >> fit and save model >> predict test >> measure accuracy

    We can easily deploy each of these stages as microservices, or Runhouse function objects containing the code
    and dependencies required for the code to run on a remote cluster.

    For the preprocessing stage, we provision a  32 CPU cluster to handle running the preprocessing and
    data splitting stages.

    For the model fitting and predict stages, we provision a GPU (in this case a A10G or A100) for our Runhouse
    functions to live.

    Notice how we pass object refs between each of the microservices - this is to prevent having to bounce around data
    between our local env and the clusters.
    """
    # Launch a new cluster (with 32 CPUs) to handle loading and processing of the dataset
    cpu = rh.cluster(name="^rh-32-cpu").up_if_not()

    raw_df = load_raw_data()
    train_data, test_data = preprocessing_and_data_split(raw_df, cpu)

    # Launch a new instance (with a GPU) to handle model training
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1', provider="cheapest").up_if_not()

    # If using AWS:
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws').up_if_not().save()

    trained_model, test_predictions = model_training(gpu, train_data, test_data)
    print(f"Saved model on gpu to path: {trained_model.path}")

    accuracy_on_gpu = rh.function(measure_accuracy, name="measure_accuracy").to(gpu, env=["pmdarima"]).save()
    accuracy = accuracy_on_gpu(test_dataset_ref=test_data, predicted_test_ref=test_predictions)
    print(f"Accuracy\n: {pickle.loads(accuracy.data)}")


if __name__ == "__main__":
    run_pipeline()
