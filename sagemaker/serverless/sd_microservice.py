import os

import runhouse as rh


def sd_generate_image(prompt):
    from diffusers import StableDiffusionPipeline

    model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base"
    ).to("cuda")
    return model(prompt).images[0]


sm_gpu_cluster = (
    rh.sagemaker_cluster(
        name="rh-sagemaker-gpu", instance_type="ml.g5.2xlarge", role=os.getenv("AWS_ROLE_ARN")
    )
    .up_if_not()
    .save()
)

# Create a Stable Diffusion microservice running on a SageMaker GPU
sd_generate = (
    rh.function(sd_generate_image)
    .to(
        sm_gpu_cluster,
        env=[
            "diffusers",
            "transformers",
        ],
    )
    .save("sd_generate")
)

img = sd_generate("A hot dog made out of matcha.")
img.show()

# To stop the instance:
# sm_gpu_cluster.teardown_and_delete()
# assert not sm_gpu_cluster.is_up()