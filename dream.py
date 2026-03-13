import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import Model
from PIL import Image
import numpy as np
import sys
import os

print("DeepDreamer")


if len(sys.argv) != 4:
    print("Usage: python3 dream.py <image> <X> <N>")
    sys.exit(1)

img_path = sys.argv[1]
layer_index = int(sys.argv[2])
steps = int(sys.argv[3])

if not os.path.exists(img_path):
    print("ERROR: Image not found")
    sys.exit(1)


base_model = MobileNetV2(weights="imagenet", include_top=False)
layer_output = base_model.layers[layer_index].output
dream_model = Model(inputs=base_model.input, outputs=layer_output)
dream_model.trainable = False


def load_image(path, target_size=(224,224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return tf.Variable(img, dtype=tf.float32)

def save_image(img_tensor, path):
    img = img_tensor.numpy()[0]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def deep_dream_step(img_var, model, step_size=0.01):
    with tf.GradientTape() as tape:
        tape.watch(img_var)
        activation = model(img_var)
        loss = tf.reduce_sum(tf.square(activation))
    gradients = tape.gradient(loss, img_var)
    gradients /= (tf.math.reduce_std(gradients) + 1e-8)
    img_var.assign_add(gradients * step_size)
    return loss


img = load_image(img_path)


octaves = 3
octave_scale = 2.0
step_size = 0.01

for octave in range(octaves):
    print(f"Octave {octave+1}/{octaves} - shape: {img.shape[1]}x{img.shape[2]}")
    for step in range(steps):
        loss = deep_dream_step(img, dream_model, step_size)
        if (step+1) % 10 == 0:
            print(f"  Step {step+1}/{steps}, loss: {loss.numpy():.4f}")
            temp_name = f"temp_o{octave}_step{step+1}.png"
            save_image(img, temp_name)
    # Upscale image for next octave
    if octave < octaves - 1:
        new_size = (int(img.shape[1]*octave_scale), int(img.shape[2]*octave_scale))
        img_up = tf.image.resize(img, new_size)
        img = tf.Variable(img_up)


out_name = "dream_" + os.path.basename(img_path)
save_image(img, out_name)
print(f"Saved DeepDream image as: {out_name}")
