import runhouse as rh
from transformers import IdeficsForVisionText2Text, AutoProcessor, TextStreamer
import torch


class IDEFICSModel(rh.Module):
    def __init__(self, model_id="HuggingFaceM4/idefics-9b-instruct", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.processor, self.model = None, None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = IdeficsForVisionText2Text.from_pretrained(self.model_id, **self.model_kwargs)

    def predict(self, prompt, stream=True, **inf_kwargs):
        if not self.model:
            self.load_model()
        inputs = self.processor(prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**inputs,
                                            streamer=TextStreamer(self.processor) if stream else None,
                                            **inf_kwargs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    gpu = rh.ondemand_cluster(name='rh-a10x', instance_type='A10G:1').up_if_not()
    env = rh.env(reqs=["transformers>=4.32.0", "accelerate>=0.21.0", "bitsandbytes>=0.41.1", "safetensors>=0.3.1",
                       "pillow", "sentencepiece", "scipy"], name="ideficsinference", working_dir="./")

    remote_idefics_model = IDEFICSModel(torch_dtype=torch.bfloat16,
                                        load_in_8bit=True,
                                        device_map='auto').get_or_to(gpu, env=env, name="idefics-model")

    prompts = [
        "Instruction: Tell me a story about this image.\n",
        "https://a-z-animals.com/media/2021/10/african-elephant-loxodonta-africana-calf-masai-mara-park-in-kenya-picture-id1262780463.jpg",
        "Answer: \n"
    ]
    test_output = remote_idefics_model.predict(prompts, max_length=100, temperature=0.7, repetition_penalty=1.0)

    print("\n\n... Full output ...\n")
    print(test_output)
    print("\n\n")
