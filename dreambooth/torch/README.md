# 🧑‍🎨 Fancy Runhouse - Dreambooth in <10 Minutes

We want Runhouse to be aggressively zero-lift. Whatever code
structure, whatever execution environment (notebook, 
orchestrator, data app, CI/CD), you should be able to do something fun
and interesting with Runhouse in minutes, not months. This tutorial
shows how you can easily import and reuse GitHub code using Runhouse.
In just a few lines of code, we can fine-tune Stable Diffusion using
Dreambooth, perform inference, and even integrate a Gradio app.

## Table of Contents
- [Dreambooth Fine-Tuning and Inference](#01-dreambooth-fine-tuning-and-inference)
- [CLIP Interrogator: Running Hugging Face Spaces](#02-clip-interrogator)
- [Appendix](#appendix)
    - [Dreambooth in Colab](#dreambooth-in-colab)
    - [Dreambooth with GitHub Functions](#dreambooth-using-github-functions)

> **Setup**:
If you haven't already, please take a look at the 
[Hardware Setup](../t01_Stable_Diffusion/README.md#00-hardware-setup) instructions in tutorial 01.

## 01 Dreambooth Fine-Tuning and Inference

Dreambooth is a popular app that lets you fine-tune Stable Diffusion on your
own images so you can personalize your Stable Diffusion inferences.
Hugging Face published a [great tutorial](https://huggingface.co/blog/dreambooth),
but it's never easy to set up on your own hardware, so various Colabs are circulating
to help people get started. We can run way faster on our own GPU, and we don't even 
need to clone down the repo! This tutorial shows how to launch a training script on your
hardware from just a GitHub URL. 

It also shows you basics of the data side of Runhouse, by:
1) Creating an `rh.folder` with the training images and then sending it to the
cluster using `folder.to(my_gpu)`. 
2) Similarly, sending the folder containing the trained model to blob storage.

This is the tip of the iceberg, and there's much more about data on the way, so
let's get started!

We present a rough walk through of the code below.
To run the full tutorial, please run locally from your laptop:
```commandline
python p01_dreambooth_train.py
python p01a_dreambooth_predict.py
```

Let's instantiate a cluster and send our local folder of training images to the
cluster. We create an `rh.folder` object that we move to the cluster, using
`folder.to(gpu)`.

We have provided some sample photos of Poppy, the company dog, in our assets folder,
or feel free to upload your own images to the folder to personalize the experience!

```python
gpu = rh.cluster(name='rh-a10x')

input_images_dir = 'assets/t02/images'
remote_image_dir = 'dreambooth/instance_images'
rh.folder(path=input_images_dir).to(system=gpu, path=remote_image_dir)
```

Now that we have the images, we can easily reuse
[Hugging Face's Dreambooth training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py),
without even needing to clone the code! First we install the necessary packages for running the script:

```python
gpu.install_packages([rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                                    install_method='pip', revision='v0.11.1'),
                     'datasets', 'accelerate', 'transformers', 'bitsandbytes',
                     'torch', 'torchvision'])
```

To run the training script, simply call `gpu.run` and input in the launch command as if you were
running directly from your command line.

```python
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
```

For an alternative way of running Dreambooth training, check out `p01b_dreambooth_train_send.py` and 
the Appendix! Runhouse also allows you to send a function to your hardware from just a GitHub URL 
pointing to the function.

Once the model is done training, we're ready for inference! Here we reuse the
`sd_generate_pinned` function from the [Stable Diffusion Tutorial](../t01_Stable_Diffusion/)
to create our `generate_dreambooth` Runhouse callable. Simply pass in the
prompt, model path, and any additional Stable Diffusion params to get results.

```python
generate_dreambooth = rh.function(fn=sd_generate_pinned, system=gpu)
my_prompt = "sks dog in a field of purple flowers"
model_path = 'dreambooth/output'
images = generate_dreambooth(my_prompt,
                             model_id=model_path,
                             num_images=4, guidance_scale=7.5,
                             steps=100)
[image.show() for image in images]
```

![](../assets/t02/p01a_output.png)

## 02 CLIP Interrogator
```commandline
python p02_gradio_clip_interrogator.py
```

Writing prompts is hard. Luckily, CLIP Interrogator can take images and generate
Stable Diffusion prompts from them. There's a popular [Hugging Face Space for CLIP 
Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator), but it'd run 
faster on our own GPU. This tutorial shows you how easy it is to take any gradio app 
and send it to your GPU, tunneled into your browser.

We start by defining a function to launch a Gradio app.

```python
def launch_gradio_space(name):
    import gradio as gr
    gr.Interface.load(name).launch()
```

As in previous tutorials, instantiate a cluster and create a Runhouse callable for
running the gradio function on the cluster.

```python
gpu = rh.cluster(name='rh-a10x')
my_space = rh.function(
    fn=launch_gradio_space,
    system=gpu,
    env=['./', 'gradio', 'fairscale', 'ftfy','huggingface-hub', 'Pillow', 
          'timm', 'open_clip_torch', 'clip-interrogator==0.3.1',]
    )
```

We can tunnel the remote Gradio space to be accessed locally using
`gpu.ssh_tunnel(local_port, remote_port)`. 

```python
gpu.ssh_tunnel(local_port=7860, remote_port=7860)
gpu.keep_warm()  # to keep the port open
```

To launch the space locally, use the `enqueue()` function.

```python
my_space.enqueue('spaces/pharma/CLIP-Interrogator')
```

The space will now be available to use at http://localhost:7860!
To stop the space, terminate the script.

>**Note**:
The first time you use the Gradio space, the model needs to download, which can
take ~10 minutes.


# Appendix

## Dreambooth in Colab

If you prefer to read or run this tutorial in Colab, you can do so 
[here](https://colab.research.google.com/github/run-house/tutorials/blob/main/t02_Dreambooth/x01_colab_dreambooth.ipynb).
This step is optional, but creating a Runhouse account allows us to 
conveniently jump into a Colab with our credentials and resources at the ready.
Check out the [docs site](https://www.run.house/docs) for more details about
[logging in](https://www.run.house/docs/tutorials/api/accessibility).

## Dreambooth using GitHub Functions

Runhouse also allows you to send a function to your hardware from just a GitHub URL pointing to the function,
without needing to clone the repo here either!

You can run this locally using:
```commandline
python p01b_dreambooth_train_send.py
```

Here, we the `main` function in 
[Hugging Face's Dreambooth training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py),
which we accomplish by directly using passing the GitHub URL and function
name to `fn` in our function object.

```python
training_function_gpu = rh.function(
    fn='https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py:main',
    system=gpu,
    env=['datasets', 'accelerate', 'transformers', 'diffusers==0.10.0',
        'torch', 'torchvision'],
    name='train_dreambooth')
```

Similarly, for creating training args using the `parse_args` function from the
same GitHub file:
```python
create_train_args = rh.function(
    fn='https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py:parse_args',
    system=gpu,
    env=[]
)
train_args = create_train_args(
    input_args=['--pretrained_model_name_or_path', 'stabilityai/stable-diffusion-2-base',
                '--instance_data_dir', remote_image_dir,
                '--instance_prompt', f'a photo of sks dog']
    )
```

Now that we have all the pieces, we can put them together as follows to train
our Dreambooth model.

```python
training_function_gpu(train_args)