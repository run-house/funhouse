import runhouse as rh

# overall class imports
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

# image paths preprocessing imports
from imutils import paths
import tensorflow as tf 
import numpy as np
import itertools

# dataset creation
import keras_cv

# dreambooth trainer
import tensorflow.experimental.numpy as tnp

# training
import math

# text
tokenizer = SimpleTokenizer() 

def process_text(caption, padding_token=49407, max_prompt_length=77):
    tokens = tokenizer.encode(caption)
    tokens = tokens + [padding_token] * (max_prompt_length - len(tokens))
    return np.array(tokens)

def encode_text(tokenized_texts, max_prompt_length):
    # import tensorflow as tf
    # from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

    POS_IDS = tf.convert_to_tensor([list(range(max_prompt_length))], dtype=tf.int32)
    text_encoder = TextEncoder(max_prompt_length)

    embedded_text = text_encoder(
        [tf.convert_to_tensor(tokenized_texts), POS_IDS], training=False
    ).numpy()

    del text_encoder
    return embedded_text

# image dataset creation
resolution = 512
augmenter = keras_cv.layers.Augmenter(
    layers=[
        keras_cv.layers.CenterCrop(resolution, resolution),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

def process_image(image_path, tokenized_text):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 3)
    image = tf.image.resize(image, (resolution, resolution))
    return image, tokenized_text

def apply_augmentation(image_batch, embedded_tokens):
    return augmenter(image_batch), embedded_tokens


def prepare_dict(instance_only=True):
    def fn(image_batch, embedded_tokens):
        if instance_only:
            batch_dict = {
                "instance_images": image_batch,
                "instance_embedded_texts": embedded_tokens,
            }
            return batch_dict
        else:
            batch_dict = {
                "class_images": image_batch,
                "class_embedded_texts": embedded_tokens,
            }
            return batch_dict
    return fn


def assemble_dataset(
    image_paths, embedded_texts, instance_only=True, batch_size=1
):  
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, embedded_texts)
    )
    dataset = dataset.map(process_image, num_parallel_calls=auto)
    dataset = dataset.shuffle(5, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(apply_augmentation, num_parallel_calls=auto)

    prepare_dict_fn = prepare_dict(instance_only=instance_only)
    dataset = dataset.map(prepare_dict_fn, num_parallel_calls=auto)
    return dataset

# To be run on GPU
def get_dreambooth_trainer(resolution, max_prompt_length, use_mp, opt_args):
    import tensorflow as tf
    from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
    from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
    from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler

    class DreamBoothTrainer(tf.keras.Model):
        # Reference: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        def __init__(
            self,
            diffusion_model,
            vae,
            noise_scheduler,
            use_mixed_precision=False,
            prior_loss_weight=1.0,
            max_grad_norm=1.0,
            **kwargs
        ):
            super().__init__(**kwargs)

            self.diffusion_model = diffusion_model
            self.vae = vae
            self.noise_scheduler = noise_scheduler
            self.prior_loss_weight = prior_loss_weight
            self.max_grad_norm = max_grad_norm

            self.use_mixed_precision = use_mixed_precision
            self.vae.trainable = False

        def train_step(self, inputs):
            instance_batch = inputs[0]
            class_batch = inputs[1]

            instance_images = instance_batch["instance_images"]
            instance_embedded_text = instance_batch["instance_embedded_texts"]
            class_images = class_batch["class_images"]
            class_embedded_text = class_batch["class_embedded_texts"]

            images = tf.concat([instance_images, class_images], 0)
            embedded_texts = tf.concat([instance_embedded_text, class_embedded_text], 0)
            batch_size = tf.shape(images)[0]

            with tf.GradientTape() as tape:
                # Project image into the latent space and sample from it.
                latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
                # Know more about the magic number here:
                # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents.
                noise = tf.random.normal(tf.shape(latents))

                # Sample a random timestep for each image.
                timesteps = tnp.random.randint(
                    0, self.noise_scheduler.train_timesteps, (batch_size,)
                )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process).
                noisy_latents = self.noise_scheduler.add_noise(
                    tf.cast(latents, noise.dtype), noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                # just the sampled noise for now.
                target = noise  # noise_schedule.predict_epsilon == True

                # Predict the noise residual and compute loss.
                timestep_embedding = tf.map_fn(
                    lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
                )
                model_pred = self.diffusion_model(
                    [noisy_latents, timestep_embedding, embedded_texts], training=True
                )
                loss = self.compute_loss(target, model_pred)
                if self.use_mixed_precision:
                    loss = self.optimizer.get_scaled_loss(loss)

            # Update parameters of the diffusion model.
            trainable_vars = self.diffusion_model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            if self.use_mixed_precision:
                gradients = self.optimizer.get_unscaled_gradients(gradients)
            gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            return {m.name: m.result() for m in self.metrics}

        def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
            half = dim // 2
            log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
            freqs = tf.math.exp(
                -log_max_preiod * tf.range(0, half, dtype=tf.float32) / half
            )
            args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
            embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
            return embedding

        def sample_from_encoder_outputs(self, outputs):
            mean, logvar = tf.split(outputs, 2, axis=-1)
            logvar = tf.clip_by_value(logvar, -30.0, 20.0)
            std = tf.exp(0.5 * logvar)
            sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
            return mean + std * sample

        def compute_loss(self, target, model_pred):
            # Chunk the noise and model_pred into two parts and compute the loss
            # on each part separately.
            # Since the first half of the inputs has instance samples and the second half
            # has class samples, we do the chunking accordingly. 
            model_pred, model_pred_prior = tf.split(model_pred, num_or_size_splits=2, axis=0)
            target, target_prior = tf.split(target, num_or_size_splits=2, axis=0)

            # Compute instance loss.
            loss = self.compiled_loss(target, model_pred)

            # Compute prior loss.
            prior_loss = self.compiled_loss(target_prior, model_pred_prior)

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
            return loss

        def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
            # Overriding this method will allow us to use the `ModelCheckpoint`
            # callback directly with this trainer class. In this case, it will
            # only checkpoint the `diffusion_model` since that's what we're training
            # during fine-tuning.
            self.diffusion_model.save_weights(
                filepath=filepath,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )

    image_encoder = ImageEncoder(resolution, resolution)
    diffusion_model = DiffusionModel(resolution, resolution, max_prompt_length)
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=opt_args['lr'],
        weight_decay=opt_args['weight_decay'],
        beta_1=opt_args['beta_1'],
        beta_2=opt_args['beta_2'],
        epsilon=opt_args['epsilon'],
    )

    dreambooth_trainer = DreamBoothTrainer(
        diffusion_model=diffusion_model,
        vae=tf.keras.Model(
                image_encoder.input,
                image_encoder.layers[-2].output,
            ),
        noise_scheduler=NoiseScheduler(),
        use_mixed_precision=use_mp,
    )
    dreambooth_trainer.compile(optimizer=optimizer, loss="mse")
    return dreambooth_trainer

# To be run on GPU
def dreambooth_train(dreambooth_trainer, ckpt_path, epochs):
    import tensorflow as tf

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    dreambooth_trainer.fit(train_dataset, epochs=epochs, callbacks=[ckpt_callback])
    return dreambooth_trainer


if __name__ == "__main__":
    # --------- Prepare Images ----------
    # Get sample instance and class images
    instance_images_root = tf.keras.utils.get_file(
        origin="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz",
        untar=True
    )
    class_images_root = tf.keras.utils.get_file(
        origin="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz",
        untar=True
    )

    instance_image_paths = list(paths.list_images(instance_images_root))
    class_image_paths = list(paths.list_images(class_images_root))

    # match instance images, ids, prompts to number of class iamges
    new_instance_image_paths = []
    for index in range(len(class_image_paths)):
        instance_image = instance_image_paths[index % len(instance_image_paths)]
        new_instance_image_paths.append(instance_image)
    
    unique_id = "sks"
    class_label = "dog"
    instance_prompts = [f"a photo of {unique_id} {class_label}"] * len(new_instance_image_paths)
    class_prompts = [f"a photo of {class_label}"] * len(class_image_paths)

    # embed prompts to save space
    max_prompt_length = 77
    tokenized_texts = np.empty((len(instance_prompts) + len(class_prompts), max_prompt_length))
    for i, caption in enumerate(itertools.chain(instance_prompts, class_prompts)):
        tokenized_texts[i] = process_text(caption, max_prompt_length=max_prompt_length)

    # Run text encoding computation on a GPU
    gpu = rh.cluster(ips=['216.153.50.68'],
                     ssh_creds={'ssh_user': 'caroline', 'ssh_private_key':'~/.ssh/id_rsa'},
                     name='coreweave-a100')    
    # gpu = rh.cluster(name='rh-cluster', instance_type='g5.2xlarge', provider='aws')
    # gpu = rh.cluster(name='rh-cluster', instance_type='A100:1', provider='gcp')
    
    # gpu = rh.cluster(name='rh-cluster', ssh_creds={'user':<user>, 'ssh_creds':{'ssh_private_key':<path_to_id_rsa>}})
    encode_text_gpu = rh.function(fn=encode_text, system=gpu, env=['tensorflow', 'keras_cv', 'imutils', 'opencv-python'])
    embedded_text = encode_text_gpu(tokenized_texts, max_prompt_length)

    # --------- Prepare Dataset ----------
    resolution = 512
    auto = tf.data.AUTOTUNE

    instance_dataset = assemble_dataset(
        new_instance_image_paths, 
        embedded_text[:len(new_instance_image_paths)],
    )
    class_dataset = assemble_dataset(
        class_image_paths, 
        embedded_text[len(new_instance_image_paths):],
        instance_only=False
    )
    train_dataset = tf.data.Dataset.zip((instance_dataset, class_dataset))

    # --------- Training ----------

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    use_mp = True # Set it to False if you're not using a GPU with tensor cores.

    # These hyperparameters come from this tutorial by Hugging Face:
    # https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
    optimizer_params = {
        'lr': 5e-6,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'weight_decay': (1e-2,),
        'epsilon': 1e-08,
    }

    get_dreambooth_trainer_gpu = rh.function(fn=get_dreambooth_trainer, system=gpu)
    dreambooth_trainer_ref = get_dreambooth_trainer_gpu.remote(resolution, max_prompt_length, use_mp, optimizer_params)
    print(f"dreambooth trainer {dreambooth_trainer_ref}")

    num_update_steps_per_epoch = train_dataset.cardinality()
    max_train_steps = 800
    epochs =  math.ceil(max_train_steps / num_update_steps_per_epoch)
    print(f"Training for {epochs} epochs.")

    ckpt_path = "dreambooth-unet.h5" 
    dreambooth_train_gpu = rh.function(fn=dreambooth_train, system=gpu)
    dreambooth_train_gpu(dreambooth_trainer_ref, ckpt_path, epochs)

    # --------- Inference ----------