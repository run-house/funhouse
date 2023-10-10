import os
import runhouse as rh
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch


class HFChatModel(rh.Module):
    def __init__(self, model_id="meta-llama/Llama-2-13b-chat-hf", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.tokenizer, self.model = None, None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, clean_up_tokenization_spaces=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **self.model_kwargs)

    def predict(self, prompt, stream=True, **inf_kwargs):
        if not self.model:
            self.load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**inputs,
                                            streamer=TextStreamer(self.tokenizer) if stream else None,
                                            **inf_kwargs)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    gpu = rh.cluster(name="rh-l4", instance_type="L4:1", autostop_mins=-1)

    env = rh.env(reqs=["transformers==4.31.0", "accelerate==0.21.0", "bitsandbytes==0.41.1", "safetensors>=0.3.1", "scipy"],
                 name="llama2inference",
                 env_vars={"GOOGLE_VM_CONFIG_LOCK_FILE": "~/.config/google-vm-config-lock.json"},
                 working_dir="./text_generation/")
    gpu.sync_secrets(["huggingface"])  # Needed to use Llama2 because it's a gated model

    remote_hf_chat_model = HFChatModel(model_id="meta-llama/Llama-2-13b-chat-hf",
                                       load_in_8bit=True,
                                       torch_dtype=torch.bfloat16,
                                       device_map='auto').to(gpu, env=env, name="llama-13b-model")

    test_prompt = "Tell me about unified development interfaces into compute and data infrastructure."
    test_output = remote_hf_chat_model.predict(test_prompt, temperature=0.7, max_new_tokens=4000, repetition_penalty=1.0)

    print("\n\n... Test Output ...\n")
    print(test_output)
