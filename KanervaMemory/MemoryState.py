import tensorflow as tf
import tensorflow_probability as tfp
from typing import NamedTuple, Tuple, Union


class MemoryState(NamedTuple):
    mean: tf.Tensor
    row_covariance: tf.Tensor

    def get_distribution(self):
        return tfp.distributions.MultivariateNormalFullCovariance(loc=self.mean,
                                                                  covariance_matrix=self.row_covariance,
                                                                  name="memory_distribution")

    def sample(self,
               sample_shape: Union[int, tf.Tensor, Tuple[int, ...], Tuple[tf.Tensor, ...]] = (),
               seed: int = None,
               name="sample_memory"):
        return self.get_distribution().sample(sample_shape, seed, name)
