import runhouse as rh

from tqdm import tqdm
import numpy as np
import hashlib
import os

# Based on https://github.com/sayakpaul/dreambooth-keras/blob/main/scripts/generate_experimental_images.py
def generate_images(
        class_images_dir="~/generated_images",
        class_prompt="a photo of a dog", 
        num_imgs_to_generate=200,
    ):
    import keras_cv
    import PIL
    from pathlib import Path

    class_images_dir = Path(class_images_dir).expanduser()
    os.makedirs(class_images_dir, exist_ok=True)
    
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
    for _ in tqdm(range(num_imgs_to_generate)):
        images = model.text_to_image(
            class_prompt,
            batch_size=3,
        )
        idx = np.random.choice(len(images))
        selected_image = PIL.Image.fromarray(images[idx])

        hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
        image_filename = os.path.join(class_images_dir, f"{hash_image}.jpg")
        selected_image.save(image_filename)
    
    return class_images_dir


if __name__ == "__main__":
    # spin up lambda cluster, using SkyPilot handling
    # use sky check to ensure provider credentials are set up correctly
    gpu = rh.cluster(name='rh-cluster', instance_type='A10:1', provider='lambda')
    # gpu = rh.cluster(name='rh-a10x')
    # gpu.restart_grpc_server()
    gpu.up_if_not()

    # byo cluster using ssh credentials
    # gpu = rh.cluster(name='rh-cluster', ssh_creds={'user':<user>, 'ssh_creds':{'ssh_private_key':<path_to_id_rsa>}})
    # gpu = rh.cluster(name='rh-cluster', ssh_creds={'user': 'ubuntu', 'ssh_creds':{'ssh_private_key': '~/.ssh/id_rsa'}})

    # Set up TF CUDA support on the cluster
    command = "conda install -y -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0; \
               mkdir -p $CONDA_PREFIX/etc/conda/activate.d; \
               echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh;"
    reqs = ['./', 'keras_cv==0.4.0', 'tensorflow>=2.10.0', 'tensorflow_datasets>=4.8.1', 'tqdm', 'numpy', 'wandb']
    # gpu.install_packages(reqs)

    # check if TF GPU support is set up properly on the cluster
    # gpu.restart_grpc_server()
    # gpu.run(['python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices(\'GPU\'))"'])

    # set up libdevice.10.bc to be discoverable by tensorflow
    gpu.run(['cp /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc .'])
    generate_images_gpu = rh.function(fn=generate_images).to(system=gpu, env=reqs)
    images_path = generate_images_gpu(num_imgs_to_generate=10)

    rh.folder(images_path, system=gpu).to('here', path='generated_images/')
