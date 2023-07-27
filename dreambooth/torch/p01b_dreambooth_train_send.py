from pathlib import Path

import runhouse as rh

# Based on https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

def train_dreambooth(input_images_dir, class_name='person'):
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    training_function_gpu = rh.function(
        fn='https://github.com/huggingface/diffusers/blob/v0.11.1/examples/dreambooth/train_dreambooth.py:main',
        system=gpu,
        env=['pip:./diffusers', 'datasets', 'accelerate', 'transformers',
              'torch', 'torchvision'],
        name='train_dreambooth')
    gpu.run_python(['import torch; torch.backends.cuda.matmul.allow_tf32 = True; '
                    'torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True'])

    remote_image_dir = 'dreambooth/instance_images'
    rh.folder(path=input_images_dir).to(system=gpu, path=remote_image_dir)

    create_train_args = rh.function(
        fn='https://github.com/huggingface/diffusers/blob/v0.11.1/examples/dreambooth/train_dreambooth.py:parse_args',
        system=gpu, env=[])
    train_args = create_train_args(input_args=['--pretrained_model_name_or_path', 'stabilityai/stable-diffusion-2-base',
                                               '--instance_data_dir', remote_image_dir,
                                               '--instance_prompt', f'a photo of sks {class_name}'])
    # train_args.train_text_encoder = True
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


if __name__ == "__main__":
    # Need about 20 photos of the subject, and the closer they can be to 512x512 the better
    train_dreambooth(input_images_dir=str(Path.home() / 'dreambooth/images'), class_name='person')
