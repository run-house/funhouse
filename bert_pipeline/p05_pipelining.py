import time
import runhouse as rh


def simple_bert_fine_tuning_service(num_epochs=3):
    """Create an end to pipeline containing the following steps:
     preprocessing: create tokenized dataset on 32 CPU cluster
     fine tuning: Fine tune the model on an A10G cluster. Model generated on the cluster will be written to s3.
     model eval: Evaluate the model accuracy on the A10G.
     deployment: Deploy the model service to the 32 CPU cluster.
     inference: Call the sentiment service with some sample data.

     Notice how easily we can create this pipeline natively in python, without the need to translate it into a
     DAG based DSL. We have the ability to easily call into services running on heterogenuous hardware simply by
     dispatching our functions to our desired compute.
    """
    # Load the dataset (for this example loading a small portion of the yelp review train data)
    dataset_run = rh.function(name="load_dataset").run("yelp_review_full", run_name="ft_pipeline", split='train[:1%]')

    # Preprocess
    preproc_table_name = dataset_run.name + "_preprocessed"
    if not rh.exists(name=preproc_table_name, resource_type='table'):
        preproc_table_name = dataset_run.name + "_preprocessed"
        preproc = rh.function(name="BERT_preproc_32cpu")
        preprocessed_table = preproc(dataset_run.name).save(name=preproc_table_name)
        print(f"Saved preprocessed table to path: {preprocessed_table.path}")
    else:
        preprocessed_table = rh.table(name=preproc_table_name)
        print(f"Loaded preprocessed table from path: {preprocessed_table.path}")

    # # Train
    model_name = dataset_run.name + '_ft_bert'
    if not rh.exists(name=model_name, resource_type='blob'):
        model_on_gpu = rh.function(name="get_model")
        optimizer_on_gpu = rh.function(name="model_optimizer")

        bert_model_run = model_on_gpu.run(run_name="bert_model", num_labels=5, model_id='bert-base-cased')
        adam_optimizer_run: rh.Run = optimizer_on_gpu.run(run_name="adam_optimizer", model=bert_model_run.name, lr=5e-5)

        ft_model = rh.function(name='finetune_ddp_1gpu')

        trained_model = ft_model(bert_model_run.name,
                                 adam_optimizer_run.name,
                                 preprocessed_table,
                                 num_epochs=num_epochs).save(model_name)

        print(f"Saved trained model to path: {trained_model.path}")
    else:
        trained_model = rh.blob(name=model_name)
        print(f"Loaded trained model from path: {trained_model.path}")

    # Evaluate
    model_eval = rh.function(name='bert_ft_eval')
    test_accuracy = model_eval(trained_model, preprocessed_table)
    print(f"Test accuracy: {test_accuracy}")

    # Deploy
    service_name = dataset_run.name + '_sa_service'
    sa_service = rh.function(name=service_name)

    return sa_service


if __name__ == "__main__":
    with rh.Run(f"bert_ft_exp_{int(time.time())}") as r:
        bert_sa_service = simple_bert_fine_tuning_service()
        print(f"Loaded sentiment analysis service from cluster: {bert_sa_service.system}")

        prompt = "I could eat hot dogs at Larry's every day."
        print(f'Review: {prompt}; sentiment score: {bert_sa_service(prompt)}')

    r.save()
