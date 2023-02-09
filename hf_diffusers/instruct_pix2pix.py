import runhouse as rh
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch


def instruct_pix2pix_generate(instruction, base_images, num_images=1,
                              steps=50, text_cfg_scale=7.5, image_cfg_scale=1.5,
                              model_id="timbrooks/instruct-pix2pix"):
    # Here we're using Runhouse's object pinning to hold the model in GPU memory. See p01a for more details.
    # We're changing the name of the pinned model to avoid a collision if reusing the cluster from p01a.
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                                      torch_dtype=torch.float16,
                                                                      safety_checker=None).to('cuda')
        rh.pin_to_memory(model_id, pipe)
    ret = []
    for image in base_images:
        # More options here: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/pix2pix
        ret = ret + pipe(instruction, image=image.resize((512, 512)),
                         guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
                         num_inference_steps=steps, num_images_per_prompt=num_images).images
    return ret


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
            'diffusers', 'accelerate', 'transformers']
    instruct_pix2pix_generate_gpu = rh.send(fn=instruct_pix2pix_generate).to(gpu, reqs=reqs)

    instruction = 'Make this into a beautiful summer painting by Claude Monet.'
    rh_base_image = Image.open('../assets/rh_logo.png').convert("RGB")

    # This takes ~3 mins to run the first time to download the model, and after that should only take ~1 sec per image.
    rh_logo_sd_images = instruct_pix2pix_generate_gpu(instruction, [rh_base_image], num_images=4,
                                                      steps=50, stream_logs=True)
    [image.show() for image in rh_logo_sd_images]
