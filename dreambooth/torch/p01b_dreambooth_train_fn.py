from pathlib import Path

import runhouse as rh

# This script shows a different style of specifying training, where you can turn existing functions in a git repo
# into your remote functions. This is useful if you're working with existing codebases, and don't want wrap your
# code in CLI boilerplate just to run it remotely.

# Based on https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    # Need about 20 photos of the subject, and the closer they can be to 512x512 the better
    input_images_dir = str(Path.home() / 'dreambooth/images')
    class_name = 'person'  # Update this to be a descriptor of the subject of your photos

    script_url = 'https://github.com/huggingface/diffusers/blob/v0.21.2/examples/dreambooth/train_dreambooth.py'
    reqs = ['pip:./diffusers', 'torch==2.0.1', 'torchvision==0.15.2', 'scipy==1.11.2',
            'datasets==2.14.5', 'accelerate==0.23.0', 'transformers==4.33.2', 'bitsandbytes==0.41.1']
    training_function_gpu = rh.function(f'{script_url}:main', name='train_dreambooth').to(gpu, env=reqs)
    gpu.run_python(['import torch; torch.backends.cuda.matmul.allow_tf32 = True; '
                    'torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True'])

    remote_image_dir = 'dreambooth/instance_images'
    rh.folder(path=input_images_dir).to(system=gpu, path=remote_image_dir)

    create_train_args = rh.function(f'{script_url}:parse_args').to(gpu, env=[])
    train_args = create_train_args(input_args=['--pretrained_model_name_or_path', 'stabilityai/stable-diffusion-2-base',
                                               '--instance_data_dir', remote_image_dir,
                                               '--instance_prompt', f'a photo of sks {class_name}'])
    # train_args.train_text_encoder = True  # For higher-memory GPUs, like an A100
    train_args.class_data_dir = 'dreambooth/class_images'
    train_args.output_dir = 'dreambooth/output'
    train_args.mixed_precision = 'bf16'
    train_args.with_prior_preservation = True
    train_args.prior_loss_weight = 1.0
    train_args.class_prompt = f"a photo of {class_name}"
    train_args.resolution = 512
    train_args.train_batch_size = 4
    train_args.gradient_checkpointing = True
    train_args.learning_rate = 1e-6
    train_args.lr_scheduler = "constant"
    train_args.lr_warmup_steps = 0
    train_args.num_class_images = 200
    train_args.checkpointing_steps = 400
    train_args.max_train_steps = 800

    training_function_gpu(train_args)
