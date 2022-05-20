import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from PIL import Image
import os

train, _ = tfds.load('cats_vs_dogs', split=['train[:90%]'], with_info=True, as_supervised=True)

SIZE = 224

print(tf.version.VERSION)


def resize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (SIZE, SIZE))
    image = image / 255.0
    return image, label


train_resized = train[0].map(resize_image)
train_batches = train_resized.shuffle(1000).batch(16)

base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
base_layers.trainable = False

model = tf.keras.Sequential([
    base_layers,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

print(model.summary())

path = "model.ckpt"
model_dir = os.path.dirname(path)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=path, save_weights_only=True, verbose=1)

image_paths = ["c" + str(i) for i in range(1, 19) if i != 16 and i != 17]
image_paths.extend(["d" + str(i) for i in range(1, 16)])

model.load_weights(path)
#  model.fit(train_batches, epochs=1, callbacks=[callback])

for __ in range(10):
    rand_img = random.choice(image_paths)
    img = Image.open("dog&&cat\\" + str(rand_img) + ".jfif")
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Կատու' if prediction < 0.5 else 'Շուն'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{pred_label} {prediction}')
    plt.show()
