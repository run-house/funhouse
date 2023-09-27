# Adapted from: https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html#tuning-the-configuration-parameters

import os
import dotenv
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

import sklearn.datasets
import sklearn.metrics

from ray import train, tune

import runhouse as rh

dotenv.load_dotenv()


def train_breast_cancer(config):
    # Load dataset
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
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
    train.report({"mean_accuracy": accuracy, "done": True})


def tune_model(config):
    tuner = tune.Tuner(
        train_breast_cancer,
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
                               instance_type="ml.m5.large").up_if_not().save()

    tune_model_remote = rh.function(tune_model, name="hp_tuning").to(cpu, env=["./", "xgboost", "sklearn", "scipy"])
    ft_results = tune_model_remote(ft_config, stream_logs=False)
    print(f"ft_results: {ft_results}")

    # For future re-use
    tune_model_remote.save()

    # TODO [JL] use Bayesian
    # TODO [JL] initialize with instance count 2 or 4 - see if Ray actually spreads it out properly
