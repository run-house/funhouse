# Runhouse & SageMaker

## 🤝 Using Runhouse with SageMaker

[Runhouse](https://www.run.house) makes the process of onboarding to SageMaker more smooth, saving you the need to 
create estimators, or conform the code to the SageMaker APIs. This translation step can take anywhere from days to 
months, and leads to rampant code duplication, forking, versioning and lineage issues.

The Runhouse [SageMakerCluster](https://www.run.house/docs/main/en/api/python/cluster#sagemakercluster-class) abstraction 
provides a few core benefits: 

* **Serverless Compute**: SageMaker provides a more serverless experience than EC2, which means you don't need to be 
responsible for auto-stopping, scheduling, or worry about spinning up and managing a K8s cluster. With SageMaker you
can easily launch multiple instances at the same time.

* **Launching with Containers**: SageMaker allows you to launch a cluster with a docker container. This gives you a 
more K8s like experience of launching compute with a lightweight image rather than an 
[AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html), which is more difficult to publish and 
expensive to maintain.

* **GPU Availability**: We've observed that GPUs tend to be more available on SageMaker compared to EC2. 

## 🚀 Getting Started

SageMaker clusters require AWS CLI V2 and configuring the SageMaker IAM role with the AWS Systems Manager.

In order to launch a cluster, you must grant SageMaker the necessary permissions with an IAM role, 
which can be provided either by name or by full ARN. You can also specify a profile explicitly or with the 
`AWS_PROFILE` environment variable.

For a more detailed walkthrough, see the
[SageMaker Hardware Setup](https://www.run.house/docs/stable/en/api/python/cluster#sagemaker-hardware-setup) section of the Runhouse docs.

## 🛣️ Usage Paths

This folder contains examples highlighting some common use cases: 

### [**Inference**](inference): Creating an inference microservice. 

Runhouse facilitates easier access to the SageMaker compute from different environments. 
You can interact with the compute from notebooks, IDEs, research, pipeline DAG, or any python interpreter. 
Runhouse allows you to SSH directly onto the cluster, update or suspend cluster autostop, and stream logs 
directly from the cluster. 

### [**Training**](training): Running a training job on a SageMaker instance. 

Let's use a simple PyTorch model to illustrate the different ways we can run training on SageMaker compute via Runhouse.
In each of these examples, Runhouse is responsible for spinning up the requested SageMaker compute, and executing the 
training code on the cluster.

(1) [**Simple train**](training/simple_train): Use Runhouse to create the SageMaker cluster and handle running the 
training code. In this example, we wrap the training code in a Runhouse function which we send to our cluster for 
execution. The changes to the source code are minimal - we simply instantiate our SageMaker cluster, wrap the training 
code in a function, and then call it in the same way we would call a local function.

(2) [**Interactive train**](training/interactive_train): Convert the training code into a Runhouse `Module` class, with 
separate methods for training, eval, and inference. While this requires slightly more modifications to the original 
source code, it gives us a stateful and interactive experience with the model on the cluster, as if we are in a 
notebook environment. We can much more easily run training epochs or try out the most recent checkpoint of the model 
that's been saved, without the need for packaging up the model and deploying it to a separate endpoint.

🦸 Both of these examples unlock a key superpower - the ability to easily run class methods on a remote cluster, 
**without** needing to translate or migrate the code onto another system.

(3) [**Train with Estimator**](training/train_with_estimator): Use the SageMaker SDK to create an
[estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html). This is useful if you are already 
using the SageMaker APIs. In this example, we define a SageMaker estimator which loads in the training code from 
a separate file. 

### [**Hyperparameter Tuning**](hyperparameter_tuning): Running a hyperparameter tuning job on a SageMaker instance. 

For this example, we use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to try different hyperparameter 
combinations on SageMaker compute. 

## 👨‍🏫 Resources
[**Docs**](https://www.run.house/docs/api/python/cluster#sagemakercluster-class):
High-level overviews of the architecture, detailed API references, and basic API examples for the SageMaker 
integration.

**Blog**: Coming soon... 

