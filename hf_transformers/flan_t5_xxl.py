import runhouse as rh
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def causal_lm_generate(prompt, model_id='google/flan-t5-xxl', **model_kwargs):
    (tokenizer, model) = rh.get_pinned_object(model_id) or (None, None)
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')
        rh.pin_to_memory(model_id, (tokenizer, model))
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, **model_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1').save()

    # 🚨 Change to flan-t5-xl if on AWS, A10G doesn't have enough GPU memory!
    # For AWS (single A100s not available, base A10G may have insufficient CPU RAM)
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')
    reqs = ['./',
            'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117',
            'transformers']
    flan_t5_generate = rh.send(fn=causal_lm_generate).to(gpu, reqs=reqs)

    # The first time this runs it will take ~20 minutes to download the model. After that it takes ~20 seconds.
    # Generation options: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
    my_prompt = "My grandmother's recipe for pasta al limone is as follows:"
    sequences = flan_t5_generate(my_prompt, max_new_tokens=1000, min_length=20, temperature=2.0, repetition_penalty=3.0,
                                 use_cache=True, do_sample=True, num_beams=2, num_return_sequences=4,
                                 stream_logs=True)

    sequences = [f"{my_prompt} {seq}" for seq in sequences]
    for seq in sequences:
        print(seq)
