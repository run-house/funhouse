# Runhouse & SageMaker

## 🤝 Using Runhouse with SageMaker

[Runhouse](https://www.run.house) makes the process of onboarding to SageMaker more smooth, saving you the need to 
create [estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html), or conform the 
code to the SageMaker APIs. This translation step can take anywhere from days to months, and leads to rampant code 
duplication, forking, versioning and lineage issues.

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

## 🛣️ Usage Paths

This folder contains examples highlighting two use cases: 

(1) [**Serverless**](inference): Creating a microservice. 

Runhouse facilitates easier access to the SageMaker compute from different environments. 
You can interact with the compute from notebooks, IDEs, research, pipeline DAG, or any python interpreter. 
Runhouse allows you to SSH directly onto the cluster, update or suspend cluster autostop, and stream logs 
directly from the cluster. 

(2) [**Training**](training): Running a dedicated training job on a SageMaker instance. 

Runhouse handles spinning up the instance and running the training job. In addition to running the job, you the 
can SSH directly into the cluster and suspend autostop even if the job has completed or failed. 

*Coming Soon*: Model inference, endpoints, batch transform jobs, and other inference related tasks.

## 👨‍🏫 Resources
[**Docs**](https://www.run.house/docs):
High-level overviews of the architecture, detailed API references, and basic API examples.

**Blog**: Coming soon... 

