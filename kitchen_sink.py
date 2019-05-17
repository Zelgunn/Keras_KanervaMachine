import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv3D, Conv3DTranspose, Reshape
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import os

from KanervaMemory import Memory


def main():
    batch_size = 16
    episode_length = 64
    width = 64
    height = 64
    memory_size = 32

    input_layer = Input([episode_length, width, height, 1])
    layer = input_layer

    layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)

    tmp_shape = layer.shape.as_list()[1:]
    code_size = tmp_shape[1] * tmp_shape[2] * tmp_shape[3]
    layer = Reshape([episode_length, code_size])(layer)

    memory = Memory(code_size=code_size, memory_size=memory_size)
    layer = memory(layer)

    layer = Reshape(tmp_shape)(layer)

    layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same")(layer)
    layer = Conv3DTranspose(filters=1, kernel_size=1, strides=1, padding="same", activation="sigmoid")(layer)

    output_layer = layer

    model = Model(inputs=input_layer,
                  outputs=output_layer)

    model.compile("adam", loss="mse", metrics=["mse"])
    model.summary()

    dataset_input_tensor = tf.random.normal(shape=[episode_length, width, height, 1])
    dataset_input_tensor = tf.clip_by_value(dataset_input_tensor, 0.0, 1.0)
    dataset = tf.data.Dataset.from_tensors(dataset_input_tensor)
    dataset = dataset.repeat(-1)
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(batch_size)

    log_dir = "../logs/KanervaMachine/log_{}".format(int(time()))
    os.makedirs(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="batch")

    model.fit(dataset, callbacks=[tensorboard], steps_per_epoch=500, epochs=100)


if __name__ == "__main__":
    main()
