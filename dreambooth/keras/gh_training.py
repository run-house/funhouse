import runhouse as rh

# Based on https://github.com/sayakpaul/dreambooth-keras/blob/main/train_dreambooth.py

if __name__ == "__main__":

    # spin up lambda cluster, using SkyPilot handling
    # use sky check to ensure provider credentials are set up correctly
    # gpu = rh.cluster(name='rh-cluster', instance_type='A100:1', provider='gcp')
    # gpu = rh.cluster(name='rh-cluster', instance_type='g5.2xlarge', provider='aws')
    gpu = rh.cluster(name='rh-cluster').up_if_not()
    gpu.restart_grpc_server()

    # byo cluster using ssh credentials
    # gpu = rh.cluster(name='rh-cluster', ssh_creds={'user':<user>, 'ssh_creds':{'ssh_private_key':<path_to_id_rsa>}})

    # pulled from https://github.com/huggingface/community-events/blob/main/keras-dreambooth-sprint/requirements.txt
    reqs = ['keras_cv==0.4.0', 'tensorflow>=2.10.0', 'tensorflow_datasets>=4.8.1',
            'pillow==9.4.0', 'imutils', 'opencv-python']
    
    create_train_args = rh.function(
        fn='https://github.com/sayakpaul/dreambooth-keras/blob/main/train_dreambooth.py:parse_args',
        system=gpu,
        env=reqs,
    )
    train_args = create_train_args()
    train_args.mp = True
    train_args.instance_images_url = 'https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz'
    train_args.class_images_url = 'https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz'

    gpu.run_python(['import tensorflow as tf', 'tf.keras.mixed_precision.set_global_policy("mixed_float16")'])
    training_function_gpu = rh.function(
        fn='https://github.com/sayakpaul/dreambooth-keras/blob/main/train_dreambooth.py:run',
        system=gpu,
        env=reqs,
        name='train_dreambooth'
    )
    training_function_gpu(train_args)
