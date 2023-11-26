# Adapted from: https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html#tuning-the-configuration-parameters

import dotenv
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from ray import tune
import xgboost as xgb

import runhouse as rh

dotenv.load_dotenv()


class TuneModel(rh.Module):
    def __init__(self):
        super().__init__()

    # Note: keeping as static since Trainable function must receive a config parameter
    @staticmethod
    def train_breast_cancer(config):
        # Load dataset
        data, labels = datasets.load_breast_cancer(return_X_y=True)
        # Split into train and test set
        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
        # Build input matrices for XGBoost
        train_set = xgb.DMatrix(train_x, label=train_y)
        test_set = xgb.DMatrix(test_x, label=test_y)
        # Train the classifier
        results = {}
        xgb.train(
            config,
            train_set,
            evals=[(test_set, "eval")],
            evals_result=results,
            verbose_eval=False,
        )
        # Return prediction accuracy
        accuracy = 1.0 - results["eval"]["error"][-1]
        tune.report({"mean_accuracy": accuracy, "done": True})

    @staticmethod
    def tune_model(config):
        tuner = tune.Tuner(
            TuneModel.train_breast_cancer,
            tune_config=tune.TuneConfig(
                num_samples=10,
            ),
            param_space=config,
        )
        res = tuner.fit()
        return res


if __name__ == "__main__":
    ft_config = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
    }

    # For a list of available instance types: https://aws.amazon.com/sagemaker/pricing/
    cpu = rh.sagemaker_cluster(name="rh-sagemaker-cpu",
                               role=os.getenv("AWS_ROLE_ARN"),
                               instance_type="ml.m5.large").save()

    tune_model_remote = TuneModel().get_or_to(cpu,
                                              env=["xgboost", "scikit-learn", "ray[tune]"],
                                              name="tune-model")
    results = tune_model_remote.tune_model(ft_config, stream_logs=False)
    print(f"ft_results: {results}")

    # For future re-use
    tune_model_remote.save()
