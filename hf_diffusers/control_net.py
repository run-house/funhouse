import runhouse as rh
# TODO: import function for control net here
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from PIL import Image
import torch

# Based on: https://github.com/lllyasviel/ControlNet

def control_net_generate(prompt, image, **model_kwargs):
    model_config_path = 'ControlNet/models/cldm_v15.yaml'
    from cldm.model import create_model, load_state_dict

    # Here we're using Runhouse's object pinning to hold the model in GPU memory.
    model = rh.get_pinned_object(model_config_path)
    if model is None:
        model = create_model(model_config_path)
        
        # TODO: properly load in the model pipeline here

        # TODO: remove the commented code below; from pix2pix
        # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
        #                                                               torch_dtype=torch.float16,
        #                                                               safety_checker=None).to('cuda')
        # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        rh.pin_to_memory(model_config_path, model)
    return model(prompt, image=image, **model_kwargs).images


if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
    gpu.up_if_not()

    # For AWS (single A100s not available, only A10G)
    # gpu = rh.cluster(name='rh-a10x', instance_type='A10G:1', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')

    reqs = ['./', 'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117',
            'diffusers', 'accelerate', 'transformers']
    
    control_net_generate_gpu = rh.function(fn=control_net_generate).to(gpu, reqs=reqs)

    prompt = "Woman running on a house."
    base_image = Image.open('../assets/rh_logo.png').convert("RGB").resize((512, 512))

    # TODO: update this comment
    # This takes ~8 mins to run the first time to download the model, and after that should only take ~2.5 sec per image.
    rh_logo_sd_images = control_net_generate_gpu(prompt, base_image,
                                                 num_inference_steps=50, num_images_per_prompt=4)
    [image.show() for image in rh_logo_sd_images]
