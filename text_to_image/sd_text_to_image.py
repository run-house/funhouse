import runhouse as rh
import torch
from diffusers import StableDiffusionPipeline


class SDModel(rh.Module):

    def __init__(self, model_id='stabilityai/stable-diffusion-2-base',
                       dtype=torch.float16, revision="fp16", device="cuda"):
        super().__init__()
        self.model_id, self.dtype, self.revision, self.device = model_id, dtype, revision, device

    @property
    def pipeline(self):
        if not hasattr(self, '_pipeline'):
            self._pipeline = StableDiffusionPipeline.from_pretrained(self.model_id,
                                                                     torch_dtype=self.dtype,
                                                                     revision=self.revision).to(self.device)
        return self._pipeline

    def predict(self, prompt, num_images=1, steps=100, guidance_scale=7.5):
        return self.pipeline(prompt, num_images_per_prompt=num_images,
                             num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x', instance_type='A10:1')
    model_gpu = SDModel().get_or_to(system=gpu, env=["torch", "diffusers", "transformers"], name="sd_model")
    my_prompt = 'A hot dog made of matcha powder.'
    images = model_gpu.predict(my_prompt, num_images=4, steps=50)
    [image.show() for image in images]

    # You can find more techniques for speeding up Stable Diffusion here:
    # https://huggingface.co/docs/diffusers/optimization/fp16
