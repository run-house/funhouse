import runhouse as rh
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig, TextStreamer
import torch


class IDEFICSModel(rh.Module):
    def __init__(self, model_id="HuggingFaceM4/idefics-9b", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.processor, self.model = None, None

    def load_model(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = IdeficsForVisionText2Text.from_pretrained(self.model_id, quantization_config=quantization_config,
                                                               **self.model_kwargs)

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
                                        device_map='auto').get_or_to(gpu, env=env, name="idefics-model")

    prompts = [
        "Instruction: provide an answer to the question. Use the image to answer.\n",
        "https://a-z-animals.com/media/2021/10/african-elephant-loxodonta-africana-calf-masai-mara-park-in-kenya-picture-id1262780463.jpg",
        "Question: What's in the picture? Answer: \n"
    ]
    test_output = remote_idefics_model.predict(prompts, max_length=50)

    print("\n\n... Test Output ...\n")
    print(test_output)
    print("\n\n")
