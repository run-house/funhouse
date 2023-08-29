# Runhouse & Airflow

## 🪄 ML Pipelines as Microservices

ML pipelines are living systems composed of smaller living systems that run repeatedly and undergo lots of iterations 
over time. Instead of slicing and packaging our pipelines into smaller scripts, 
we should ideally modularize them into small living microservices.

[Runhouse](https://www.run.house) provides a low lift and migration free alternative to building ML pipelines. 
By being aggressively DSL free, Runhouse requires the same activation energy to deploy a 
microservice on remote compute as it does to create one, effectively eliminating the division
between “code” and “workflow” and creating a much more pleasant DevX.

## 🤝 Integrating Runhouse with Airflow

Runhouse is a unified interface into *existing* compute and data systems, built to reclaim
the 50-75% of ML practitioners' time lost to debugging, adapting, or repackaging code
for different environments.

By using Airflow to orchestrate Runhouse code, we save the extra translation step required by
Airflow (and other orchestration tools), to break up existing code into the "glue code" required for each
task. By not having to translate to the code into a DSL, we keep the code in its original form,
ensuring re-usability and saving us this extra translation step while still getting the benefits of
the orchestration tool (scheduling, triggering, monitoring, fault tolerance, etc.).

## 📓 Examples

This folder contains three examples: 
* [**Airflow**](https://github.com/deliveryhero/pyconde2019-airflow-ml-workshop/blob/be138e85b0a2658988e4e57b9432bd27b089a8fe/dags/prediction_pipeline.py): Setting up a worfklow with Airflow.
* [**Runhouse**](runhouse_pipeline): Setting up a worfklow with Runhouse.
* [**Airflow & Runhouse**](airflow_and_runhouse): Integrating Runhouse into an existing Airflow workflow. 

## 👨‍🏫 Resources
[**Blog**](https://www.run.house/blog/supercharging-airflow-with-runhouse): A detailed explanation of how to integrate 
Runhouse with Airflow, and how using both packages together gives us the best of both worlds: A python-native solution 
for defining workflows across heterogenous compute, and a robust orchestration tool for scheduling, triggering and 
monitoring workflows.

[**Docs**](https://www.run.house/docs/stable/en):
High-level overviews of the architecture, detailed API references, and basic API examples.
