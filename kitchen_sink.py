from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

from Memory import Memory


def main():
    batch_size = 6
    episode_length = 2
    code_size = 100
    memory_size = 32

    memory = Memory(code_size=code_size,
                    memory_size=memory_size,
                    batch_size=batch_size)
    memory_prior_state = memory.get_prior_state(batch_size)

    input_layer = Input([episode_length, code_size])
    layer = memory(input_layer, prior_state=memory_prior_state)

    output_layer = layer

    model = Model(inputs=input_layer,
                  outputs=output_layer)

    model.compile("adam", loss="mse")
    model.summary()


if __name__ == "__main__":
    main()
