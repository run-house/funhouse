import runhouse as rh

# Based on https://github.com/sayakpaul/dreambooth-keras/blob/main/train_dreambooth.py

if __name__ == "__main__":
    # spin up lambda cluster, using SkyPilot handling
    # use sky check to ensure provider credentials are set up correctly
    gpu = rh.cluster(name='rh-cluster', instance_type='A100:1', provider='lambda')
    gpu.up_if_not()

    # byo cluster using ssh credentials
    # gpu = rh.cluster(name='rh-cluster', ssh_creds={'user':<user>, 'ssh_creds':{'ssh_private_key':<path_to_id_rsa>}})

    # pulled from https://github.com/huggingface/community-events/blob/main/keras-dreambooth-sprint/requirements.txt
    gpu.install_packages([
            rh.GitPackage(git_url='https://github.com/sayakpaul/dreambooth-keras.git', install_method='local'),
            'keras_cv==0.4.0',
            'tensorflow>=2.10.0',
            'tensorflow_datasets>=4.8.1',
            'pillow==9.4.0',
            'imutils',
            'opencv-python',
            'wandb',
    ])

    instance_images_url = 'https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz'
    class_images_url = 'https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz'

    gpu.run([f'python dreambooth-keras/train_dreambooth.py --mp '
             f'--instance_images_url={instance_images_url} '
             f'--class_images_url={class_images_url}'])
