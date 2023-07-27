import runhouse as rh
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import torch

# Based on: https://github.com/timothybrooks/instruct-pix2pix

def instruct_pix2pix_generate(instruction, image, **model_kwargs):
    model_id = "timbrooks/instruct-pix2pix"
    # Here we're using Runhouse's object pinning to hold the model in GPU memory.
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                                      torch_dtype=torch.float16,
                                                                      safety_checker=None).to('cuda')
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        rh.pin_to_memory(model_id, pipe)
    return pipe(instruction, image=image, **model_kwargs).images


if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')

    # For AWS (single A100s not available, only A10G)
    # gpu = rh.cluster(name='rh-a10x', instance_type='A10G:1', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')

    reqs = ['./', 'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117',
            'diffusers', 'accelerate', 'transformers']
    instruct_pix2pix_generate_gpu = rh.function(fn=instruct_pix2pix_generate).to(gpu, reqs=reqs)

    instruction = "Make this into an illustration for a children's book."
    base_image = Image.open('../_assets/rh_logo.png').convert("RGB").resize((512, 512))

    # This takes ~8 mins to run the first time to download the model, and after that should only take ~2.5 sec per image.
    rh_logo_sd_images = instruct_pix2pix_generate_gpu(instruction, base_image,
                                                      num_inference_steps=50, num_images_per_prompt=4)
    [image.show() for image in rh_logo_sd_images]
