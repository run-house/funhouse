import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch


def sd_generate_txt_to_img(prompt, num_images=1, steps=100, guidance_scale=7.5,
                           model_id='stabilityai/stable-diffusion-2-base',
                           dtype=torch.float16, revision="fp16"):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, revision=revision).to("cuda")
        rh.pin_to_memory(model_id, pipe)
    return pipe(prompt, num_images_per_prompt=num_images,
                num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')

    # For AWS (single A100s not available, base A10G may have insufficient CPU RAM)
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')
    reqs = ['./',
            'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117',
            'diffusers',
            'transformers']
    generate_gpu = rh.function(fn=sd_generate_txt_to_img).to(gpu, reqs=reqs)
    my_prompt = 'A hot dog made of matcha powder.'
    images = generate_gpu(my_prompt, num_images=4, steps=50)
    [image.show() for image in images]

    # You can find more techniques for speeding up Stable Diffusion here:
    # https://huggingface.co/docs/diffusers/optimization/fp16
