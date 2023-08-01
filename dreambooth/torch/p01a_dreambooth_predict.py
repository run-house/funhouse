import runhouse as rh

from t01_Stable_Diffusion.p02_faster_sd_generate import sd_generate_pinned

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    generate_dreambooth = rh.function(fn=sd_generate_pinned).to(gpu)
    my_prompt = "sks dog in a field of purple flowers"
    model_path = 'dreambooth/output'
    images = generate_dreambooth(my_prompt,
                                 model_id=model_path,
                                 num_images=4, guidance_scale=7.5,
                                 steps=100)
    [image.show() for image in images]
