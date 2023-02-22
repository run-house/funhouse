import runhouse as rh
from diffusers import UnCLIPPipeline
import torch


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
    generate_karlo_gpu = rh.function(fn=unclip_generate_img).to(gpu, reqs=reqs)

    # The first time we run this it will take ~8 minutes to download the model, which is pretty large.
    # Subsequent calls will only take ~1 second per image
    my_prompt = 'beautiful fantasy painting of Tom Hanks as Samurai in sakura field'
    images = generate_karlo_gpu(my_prompt, num_images=4)
    [image.show() for image in images]