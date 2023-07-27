import runhouse as rh
import torch
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline

# Based on: https://huggingface.co/stabilityai/sd-x2-latent-upscaler


def sd_generate(
    prompt, num_images=1, model_id="stabilityai/stable-diffusion-2-base", **model_kwargs
):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained( model_id, torch_dtype=torch.float16, revision="fp16").to("cuda")
        rh.pin_to_memory(model_id, pipe)
    return pipe(prompt, num_images_per_prompt=num_images, **model_kwargs).images


def sd_latent_upscaler(
    prompt, image, model_id="stabilityai/sd-x2-latent-upscaler", **model_kwargs
):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        rh.pin_to_memory(model_id, pipe)
    return pipe(prompt, image=image, **model_kwargs).images


if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    # gpu = rh.cluster(name='rh-a10x', instance_type='A100:1').save()
    gpu = rh.cluster(name='rh-a10x')

    # For AWS (single A100s not available, only A10G)
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')


    reqs = ["./",  "torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117",
            "diffusers", "transformers"]
    
    prompt = "sad mark zuckerberg."
    num_imgs = 4

    # generate images using stable diffusion
    generate_gpu = rh.function(fn=sd_generate).to(gpu, reqs=reqs)
    images = generate_gpu(prompt, num_images=num_imgs, num_inference_steps=100, guidance_scale=7.5)
    [image.show() for image in images]

    # upscale the images generated above
    latent_upscaler_gpu = rh.function(fn=sd_latent_upscaler).to(gpu, reqs=reqs)
    upscaled_image = latent_upscaler_gpu([prompt] * num_imgs, image=images, num_inference_steps=20, guidance_scale=0)
    [image.show() for image in upscaled_image]
