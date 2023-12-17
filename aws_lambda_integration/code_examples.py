import runhouse as rh
from enum import Enum
import boto3
from diffusers import UnCLIPPipeline
import torch


# -------------------  Basic function creation -------------------
class MathAction(Enum):
    ADD = "+"
    SUBSTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"


def basic_calculator(a: [int, float], b: [int, float], action: MathAction):
    if action.ADD:
        return a + b
    elif action.SUBSTRACT:
        return a - b
    elif action.MULTIPLY:
        return a * b
    else:
        if b == 0:
            raise ZeroDivisionError
        return a / b


# from callable function
my_lambda_calc = rh.aws_lambda_fn(basic_calculator, name="my_first_lambda")
my_lambda_calc.save()  # save config to Runhouse den.
addition_res = float(my_lambda_calc(1.2, 4, "+"))  # execute the function in aws.
multiply_res = int(my_lambda_calc(435, 543, "*"))  # execute the function in aws.
my_lambda_calc.share(users=["josh@run.house"])  # sharing the function with another team member (user)
my_reloaded_lambda = rh.aws_lambda_fn(name="my_first_lambda")  # reloading the function
my_lambda_calc.delete()  # deleting the resource from AWS and Runhouse den.

# from rh.function instance
lambda_from_rh_func = rh.function(basic_calculator).to("Lambda_Function")
lambda_from_rh_func.save()
calc_map = lambda_from_rh_func.map([5, 6, 7, 8], [1, 2, 3, 4], ["+", "-", "*", "/"])  # ["6", "4", "21", "2"]
calc_starmap = lambda_from_rh_func.starmap((5, 1, "+"), (6, 2, "-"), (7, 3, "*"), (8, 4, "/"))  # ["6", "4", "21", "2"]


# -------------------  Preprocessing function -------------------
# from .py file
ssm_client = boto3.client('ssm')
hf_token = ssm_client.get_parameter(
    Name='/huggingFace/sasha/token',
    WithDecryption=True
)
lambda_env = rh.env(reqs=["pandas", "huggingface_hub"], env_vars={'hf_token': hf_token}, name="lambda_env")
preprocess_lambda = rh.aws_lambda_fn(
    paths_to_code="/Users/sashabelousovrh/PycharmProjects/AWS_lambda_poc_current/runhouse/tests/test_resources"+
                  "/test_modules/test_functions/test_aws_lambda/test_helpers/lambda_tests/preprocess_demo.py",
    handler_function_name="preprocess",
    runtime="python3.9",  # optional
    name="preprocess_openbookqa",  # optional
    env=lambda_env  # optional, in this case nessecary since need to install some packagrs as well as set the token as an env_var
)
processed_data = preprocess_lambda("openbookqa", "train", "additional")
print(processed_data.head(10))
print(f'size after preprocessing is {processed_data.shape[0]}')


# download podcasts


# ------------------- models loading and usage (Hugging Face) -------------------

def unclip_generate_img(prompt,
                        model_id='kakaobrain/karlo-v1-alpha',
                        num_images=1,
                        **model_kwargs):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = UnCLIPPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
        rh.pin_to_memory(model_id, pipe)
    return pipe([prompt], num_images_per_prompt=num_images, **model_kwargs).images


if __name__ == "__main__":
    reqs = ['torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117',
            'diffusers',
            'transformers']
    generate_karlo_lambda = rh.aws_lambda_fn(fn=unclip_generate_img, env=reqs).to()

    # The first time we run this it will take ~8 minutes to download the model, which is pretty large.
    # Subsequent calls will only take ~1 second per image
    my_prompt = 'beautiful fantasy painting of Tom Hanks as Samurai in sakura field'
    images = generate_karlo_lambda(my_prompt, num_images=4)
    [image.show() for image in images]