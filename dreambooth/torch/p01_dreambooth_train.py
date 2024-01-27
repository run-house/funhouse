import runhouse as rh

# Based on https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

if __name__ == "__main__":
    input_images_dir = 'assets/t02/images'
    class_name = 'dog'
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    gpu.up_if_not()
    gpu.install_packages([rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                                        install_method='pip', revision='v0.11.1'),
                          'datasets', 'accelerate', 'transformers', 'bitsandbytes', 'torch', 'torchvision'])

    rh.folder(path=input_images_dir).to(system=gpu, path='dreambooth/instance_images')

    gpu.run([f'accelerate launch diffusers/examples/dreambooth/train_dreambooth.py '
             f'--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-base '
             f'--instance_data_dir=dreambooth/instance_images '
             f'--class_data_dir=dreambooth/class_images '
             f'--output_dir=dreambooth/output '
             f'--with_prior_preservation --prior_loss_weight=1.0 '
             f'--instance_prompt="a photo of sks {class_name}" '
             f'--class_prompt="a photo of {class_name}" '
             f'--resolution=512 --max_train_steps=800 '
             f'--train_batch_size=1 --gradient_accumulation_steps=2 --gradient_checkpointing --use_8bit_adam '
             f'--learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=200 '
             f'--mixed_precision=bf16 '
             # f'--train_text_encoder '  # Uncomment if training on A100, but too heavy for A10G (AWS)
             ])
