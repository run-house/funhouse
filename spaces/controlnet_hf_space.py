import runhouse as rh


def launch_gradio_space(space_name):
    import gradio as gr  # If we put this inside the function, it doesn't even need to be installed locally
    gr.Interface.load("spaces/" + space_name).launch()


# Based on https://huggingface.co/spaces/hysts/ControlNet

if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')

    # For AWS (single A100s not available, base A10G may have insufficient CPU RAM)
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-a10x')

    reqs = ['./',
            'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117',
            'transformers', 'gradio', 'addict', 'albumentations', 'einops', 'imageio', 'imageio-ffmpeg',
            'kornia', 'omegaconf', 'open-clip-torch', 'opencv-contrib-python', 'opencv-python-headless',
            'prettytable', 'pytorch-lightning', 'safetensors', 'xformers', 'yapf',
            'huggingface-hub', 'timm', 'torchvision']
    my_space = rh.function(fn=launch_gradio_space).to(gpu, reqs=reqs)

    gpu.ssh_tunnel(local_port=7860, remote_port=7860)
    gpu.keep_warm()
    run_key = None
    try:
        run_key = my_space.remote("hysts/ControlNet")
        gpu.get(run_key, stream_logs=True)
        # To stop the space, just terminate this script.
    except KeyboardInterrupt as e:
        gpu.cancel(run_key)
