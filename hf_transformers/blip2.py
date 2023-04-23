import requests
import runhouse as rh
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def blip2_img_to_txt(
    images,
    prompts: list[str | None] = None,
    max_new_tokens=20,
    model_id="Salesforce/blip2-opt-2.7b",
    dtype=torch.float16,
) -> str:
    model = rh.get_pinned_object(model_id)
    if model is None:
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to("cuda")
        rh.pin_to_memory(model_id, model)

    processor = rh.get_pinned_object(f"{model_id}_processor")
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_id)
        rh.pin_to_memory(f"{model_id}_processor", processor)

    inputs = processor(images=images, text=prompts, return_tensors="pt").to(
        "cuda", dtype
    )
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    stripped_text = [text.strip() for text in generated_texts]

    return stripped_text


if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    # gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')

    # For AWS (single A100s not available, base A10G may have insufficient CPU RAM)
    gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')

    reqs = [
        "./",
        "torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117",
        "transformers",
    ]

    img_to_txt_gpu = rh.function(fn=blip2_img_to_txt).to(gpu, reqs=reqs)

    urls = [
        "https://www.cookwithmanali.com/wp-content/uploads/2020/05/Masala-Dosa.jpg",
        "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png",
    ]

    images = [
        Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in urls
    ]

    captions = img_to_txt_gpu(images, prompts=None)
    print(captions)
