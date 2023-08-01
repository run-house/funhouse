import runhouse as rh

def launch_gradio_space(name):
    import gradio as gr  # If we put this inside the function, it doesn't even need to be installed locally
    gr.Interface.load(name).launch()

# Based on https://huggingface.co/spaces/pharma/CLIP-Interrogator/

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    my_space = rh.function(fn=launch_gradio_space).to(gpu, env=['./', 'gradio', 'fairscale', 'ftfy',
                                                             'huggingface-hub', 'Pillow', 'timm',
                                                             'open_clip_torch', 'clip-interrogator==0.3.1'])
    gpu.ssh_tunnel(local_port=7860, remote_port=7860)
    gpu.keep_warm()
    run_key = None
    try:
        print('The first time you run the space it needs to download the model, which can take ~10 minutes.')
        run_key = my_space.remote('spaces/pharma/CLIP-Interrogator')
        gpu.get(run_key, stream_logs=True)
        # To stop the space, just terminate this script.
    except KeyboardInterrupt as e:
        gpu.cancel(run_key)
