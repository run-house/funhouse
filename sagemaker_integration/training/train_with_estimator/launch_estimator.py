import os
import dotenv
import runhouse as rh
from sagemaker.pytorch import PyTorch

dotenv.load_dotenv()

# https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase
estimator = PyTorch(
    entry_point="train.py",
    # Estimator requires a role ARN (can't be a profile)
    role=os.getenv("AWS_ROLE_ARN"),
    # Script can sit anywhere in the file system
    source_dir=os.path.abspath(os.getcwd()),
    framework_version="2.1.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.m5.large",
    # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
    keep_alive_period_in_seconds=3600,
    # A list of absolute or relative paths to directories with any additional libraries that
    # should be exported to the cluster
    dependencies=[],
)

cluster_name = "rh-sagemaker-training"
c = rh.sagemaker_cluster(name=cluster_name, estimator=estimator)
c.save()

# To stop the training job:
# reloaded_cluster.teardown_and_delete()
# assert not reloaded_cluster.is_up()
