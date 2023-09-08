import runhouse as rh
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    gpu = rh.ondemand_cluster(name='rh-a10x', instance_type='A10:1')
    GPUModel = rh.module(SentenceTransformer).to(gpu, env=["torch", "transformers", "sentence-transformers"])
    model = GPUModel("sentence-transformers/all-mpnet-base-v2", device="cuda")
    text = ["Keanu Reeves is an American treasure."]
    print(model.encode(text))
