import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.initializers import constant, zeros, truncated_normal
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras import backend
from tensorflow_probability.python.distributions import MultivariateNormalDiag
import functools
from typing import Optional, Dict, Any, Union, Tuple, Callable

from KanervaMemory.MemoryState import MemoryState


def define_scope(scope: str = None):
    def decorator(function):
        name_scope = scope or function.__name__

        @functools.wraps(function)
        def decorated(*args, **kwargs):
            with tf.name_scope(name_scope):
                return function(*args, **kwargs)

        return decorated

    return decorator


class Memory(Layer):
    def __init__(self,
                 code_size: int,
                 memory_size: int,
                 iteration_count: int = 1,
                 w_prior_stddev: float = 1.0,
                 initial_w_stddev: float = 0.3,
                 observational_noise_stddev: float = 1.0,
                 use_addressing_matrix=False,
                 use_memory_mean_as_samples=True,
                 use_w_mean_as_samples=True,
                 batch_size: int = None,
                 **kwargs):
        super(Memory, self).__init__(**kwargs)

        self.code_size = code_size
        self.memory_size = memory_size
        self.iteration_count = iteration_count
        self.w_prior_stddev = w_prior_stddev
        self.initial_w_stddev = initial_w_stddev
        self.observational_noise_stddev = observational_noise_stddev
        self.use_addressing_matrix = use_addressing_matrix
        self.use_memory_mean_as_samples = use_memory_mean_as_samples
        self.use_w_mean_as_samples = use_w_mean_as_samples
        self.batch_size: Optional[int] = batch_size

        self.prior_memory_mean: Optional[tf.Tensor] = None
        self.prior_memory_covariance: Optional[tf.Tensor] = None
        self.w_prior_distribution: Optional[MultivariateNormalDiag] = None

        self._code_size: Optional[tf.Tensor] = None
        self._memory_size: Optional[tf.Tensor] = None
        self._iteration_count: Optional[tf.Tensor] = None
        self._batch_size: Optional[tf.Tensor] = None

        self._w_prior_stddev: Optional[tf.Tensor] = None
        self._w_stddev: Optional[tf.Tensor] = None
        self._observational_noise_stddev: Optional[tf.Tensor] = None
        self._w_distribution: Optional[MultivariateNormalDiag] = None

        self.input_spec = InputSpec(shape=[None, None, self.code_size])

    def build(self, input_shape: TensorShape):
        if self.prior_memory_mean is None:
            self.build_prior_state()

        self._code_size = tf.constant(self.code_size, name="code_size")
        self._memory_size = tf.constant(self.memory_size, name="memory_size")
        self._iteration_count = tf.constant(self.iteration_count, name="iteration_count")
        if self.batch_size is not None:
            self._batch_size = tf.constant(self.batch_size, name="batch_size")

        # region Address weights
        with tf.name_scope("w_prior"):
            self._w_prior_stddev = tf.constant(self.w_prior_stddev,
                                               name="w_prior_stddev")
    
            self.w_prior_distribution = MultivariateNormalDiag(loc=tf.zeros(shape=[self._memory_size]),
                                                               scale_identity_multiplier=self._w_prior_stddev,
                                                               name="w_prior_distribution")

        log_w_stddev = self.add_weight(initializer=constant(self.initial_w_stddev),
                                       name="log_w_stddev",
                                       shape=[])
        self._w_stddev = tf.exp(log_w_stddev, name="w_stddev")
        # endregion

        # region Observational noise
        if self.observational_noise_stddev > 0.0:
            observational_noise_stddev = tf.constant(self.observational_noise_stddev,
                                                     name="observational_noise_stddev")
        else:
            log_observational_noise_stddev = self.add_weight(initializer=zeros(),
                                                             name="log_observational_noise_stddev",
                                                             shape=[])
            observational_noise_stddev = tf.exp(log_observational_noise_stddev,
                                                name="observational_noise_stddev")
        self._observational_noise_stddev = observational_noise_stddev
        # endregion

        self.built = True

    def build_prior_state(self):
        with tf.name_scope("prior_state"):
            # region Prior memory mean
            mean_initializer = truncated_normal(mean=0.0, stddev=1.0)
            self.prior_memory_mean = self.add_weight(name="prior_memory_mean",
                                                     shape=[self.memory_size, self.code_size],
                                                     initializer=mean_initializer)
            # endregion

            # region Prior memory covariance
            log_variance_scale = self.add_weight(name="prior_memory_log_variance_scale",
                                                 shape=[],
                                                 initializer=zeros)
            variance = log_variance_scale * tf.ones([self.memory_size]) + backend.epsilon()
            self.prior_memory_covariance = tf.matrix_diag(variance, name="prior_memory_covariance")
            # endregion

        self._non_trainable_weights += [self.prior_memory_covariance, self.prior_memory_mean]

    def compute_output_shape(self, input_shape):
        print("Memory.compute_output_shape - input_shape:type", type(input_shape))
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        if "prior_state" in kwargs:
            prior_state = kwargs["prior_state"]
            if not isinstance(prior_state, MemoryState):
                raise TypeError
        else:
            batch_size = tf.shape(inputs)[0]
            prior_state = self.get_prior_state(batch_size)

        # TODO : Remove need for transpose !
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        posterior_memory, w_divergence_episode, memory_divergence_episode = self.update_state(prior_state, inputs)

        z_mean, w_divergence = self.read_memory(posterior_memory, inputs)

        z_distribution = self.get_z_distribution(z_mean)
        z_sample = z_distribution.sample(name="sample_z")

        with tf.name_scope("total_kl_divergence"):
            total_divergence = tf.reduce_mean(w_divergence_episode + w_divergence)
        self.add_loss(total_divergence)

        z_sample = tf.transpose(z_sample, perm=[1, 0, 2])

        return z_sample

    @define_scope()
    def get_prior_state(self, batch_size: Union[int, tf.Tensor]) -> MemoryState:
        """Return the prior distribution of memory as a MemoryState."""

        if self.prior_memory_mean is None:
            self.build_prior_state()

        # region Tiling prior Memory to match Batch size
        mean = tf.expand_dims(self.prior_memory_mean, axis=0)
        covariance = tf.expand_dims(self.prior_memory_covariance, axis=0)

        mean = tf.tile(mean, [batch_size, 1, 1])
        covariance = tf.tile(covariance, [batch_size, 1, 1])
        # endregion

        return MemoryState(mean=mean,
                           row_covariance=covariance)

    # region Update memory state
    @define_scope()
    def update_state(self,
                     memory_state: MemoryState,
                     z: tf.Tensor
                     ) -> Tuple[MemoryState, tf.Tensor, tf.Tensor]:
        if self.use_addressing_matrix:
            raise NotImplementedError

        z_shape = tf.shape(z)
        episode_length = z_shape[0]

        # region Loop Vars (initial values)
        initial_memory = memory_state
        initial_i = tf.constant(0, name="episode_loop_initial_i")

        divergence_elements_shape = None if self.batch_size is None else [1, self.batch_size]
        # w_array = tf.TensorArray(dtype=backend.floatx,
        #                          size=episode_length,
        #                          element_shape=divergence_elements_shape + [self.memory_size],
        #                          name="w_array")

        w_divergence_array = tf.TensorArray(dtype=backend.floatx(),
                                            size=episode_length,
                                            element_shape=divergence_elements_shape,
                                            name="w_divergence_array")

        memory_divergence_array = tf.TensorArray(dtype=backend.floatx(),
                                                 size=episode_length,
                                                 element_shape=divergence_elements_shape,
                                                 name="memory_divergence_array")

        loop_vars = (initial_i, initial_memory, w_divergence_array, memory_divergence_array)

        # loop_vars = (initial_i, initial_memory, w_array, w_divergence_array, memory_divergence_array)
        # endregion

        def loop_cond(i, _, __, ___):
            return i < episode_length

        def loop_body(i, step_memory, w_step_divergence_array, memory_step_divergence_array):
            z_step = tf.expand_dims(z[i], axis=0)
            new_memory, w_step_mean, w_step_sample = self.update_state_optimization(step_memory, z_step)
            w_step_divergence = self.get_w_divergence(w_step_mean)
            memory_step_divergence = self.get_update_divergence(step_memory, w_step_sample, z_step)
            return (i + 1,
                    new_memory,
                    # w_step_array.write(i, w_step_sample),
                    w_step_divergence_array.write(i, w_step_divergence),
                    memory_step_divergence_array.write(i, memory_step_divergence))

        loop_outputs = tf.while_loop(cond=loop_cond,
                                     body=loop_body,
                                     loop_vars=loop_vars)
        # region Process loop outputs
        loop_outputs: Tuple[tf.Tensor, MemoryState, tf.TensorArray, tf.TensorArray] = loop_outputs

        _, posterior_memory, w_divergence_array, memory_divergence_array = loop_outputs
        # _, posterior_memory, w_array, w_divergence_array, memory_divergence_array = loop_outputs

        # w_episode = w_array.concat(name="w_episode")
        w_divergence_episode = w_divergence_array.concat(name="w_divergence_episode")
        memory_divergence_episode = memory_divergence_array.concat(name="memory_divergence_episode")
        # endregion

        return posterior_memory, w_divergence_episode, memory_divergence_episode

    def update_state_optimization(self,
                                  memory_state: MemoryState,
                                  z: tf.Tensor
                                  ) -> Tuple[MemoryState, tf.Tensor, tf.Tensor]:
        initial_memory = memory_state
        initial_i = tf.constant(0, name="optimization_loop_initial_i")

        loop_vars = (initial_i, initial_memory)

        def loop_cond(i, _):
            return i < (self._iteration_count - tf.constant(1))

        def update_step(step_memory):
            memory_sample = self.sample_memory(step_memory)
            w_step_mean = self.solve_w_mean(memory_sample, z)
            step_memory = self.update_memory(initial_memory, w_step_mean, z)
            return step_memory, w_step_mean

        def loop_body(i, step_memory):
            step_memory, _ = update_step(step_memory)
            return i + 1, step_memory

        _, memory_state = tf.while_loop(cond=loop_cond,
                                        body=loop_body,
                                        loop_vars=loop_vars)

        memory_state, w_mean = update_step(memory_state)
        w_sample = self.sample_w(w_mean)

        return memory_state, w_mean, w_sample

    def update_memory(self, memory_state: MemoryState,
                      w_sample: tf.Tensor,
                      z_mean: tf.Tensor
                      ) -> MemoryState:
        initial_memory_mean, initial_memory_row_covariance = memory_state

        predicted_z_mean = self.z_mean_from_memory(w_sample, initial_memory_mean, name="predicted_z")
        prediction_delta = z_mean - predicted_z_mean

        cross_variance = tf.einsum("ebm,bmn->ebn", w_sample, initial_memory_row_covariance,
                                   name="cross_variance_between_memory_and_w")
        variance_noise = tf.eye(tf.shape(cross_variance)[0], name="variance_noise_diagonal_matrix")
        z_episode_covariance = tf.einsum("ebm,ebm->eb", cross_variance, w_sample, name="z_episode_covariance")
        z_episode_covariance += variance_noise

        tmp = cross_variance / tf.expand_dims(z_episode_covariance, axis=-1)
        posterior_mean = initial_memory_mean + tf.einsum("ebm,ebc->bmc", tmp, prediction_delta,
                                                         name="posterior_memory_mean")
        posterior_row_covariance = initial_memory_row_covariance - tf.einsum("ebm,ebn->bmn", tmp, cross_variance,
                                                                             name="posterior_memory_row_covariance")
        # region Clip for numerical stability
        posterior_row_covariance_diagonal = tf.matrix_diag_part(posterior_row_covariance)
        posterior_row_covariance_diagonal = tf.clip_by_value(posterior_row_covariance_diagonal,
                                                             clip_value_min=backend.epsilon(),
                                                             clip_value_max=1e10)
        posterior_row_covariance = tf.matrix_set_diag(posterior_row_covariance, posterior_row_covariance_diagonal)
        # endregion
        posterior_memory = MemoryState(mean=posterior_mean,
                                       row_covariance=posterior_row_covariance)
        return posterior_memory

    # endregion

    # region Read from memory with z
    @define_scope()
    def read_memory(self,
                    memory_state: MemoryState,
                    z: tf.Tensor,
                    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.use_addressing_matrix:
            return self.read_memory_with_addressing_matrix(memory_state, z)
        else:
            return self.read_memory_without_addressing_matrix(memory_state, z)

    def read_memory_without_addressing_matrix(self,
                                              memory_state: MemoryState,
                                              z: tf.Tensor,
                                              ) -> Tuple[tf.Tensor, tf.Tensor]:
        memory_sample = self.sample_memory(memory_state)
        w_mean = self.solve_w_mean(memory_sample, z)
        w_sample = self.sample_w(w_mean)
        z_mean = self.z_mean_from_memory(w_sample, memory_sample)
        w_divergence = self.get_w_divergence(w_mean)
        return z_mean, w_divergence

    def read_memory_with_addressing_matrix(self,
                                           memory_state: MemoryState,
                                           z: tf.Tensor,
                                           ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    # endregion

    # region Distributions
    def get_w_distribution(self,
                           w_mean
                           ) -> MultivariateNormalDiag:
        return MultivariateNormalDiag(loc=w_mean,
                                      scale_identity_multiplier=self._w_stddev,
                                      name="w_distribution")

    def get_z_distribution(self,
                           z_mean: tf.Tensor
                           ) -> MultivariateNormalDiag:
        return MultivariateNormalDiag(loc=z_mean,
                                      scale_identity_multiplier=self._observational_noise_stddev,
                                      name="z_distribution")

    # endregion

    # region Sampling
    @define_scope()
    def sample_memory(self, memory_state: MemoryState) -> tf.Tensor:
        if self.use_memory_mean_as_samples:
            return memory_state.mean
        else:
            return memory_state.sample(self._code_size)

    @define_scope()
    def sample_w(self, w_mean: tf.Tensor) -> tf.Tensor:
        if self.use_w_mean_as_samples:
            return w_mean
        else:
            return self.get_w_distribution(w_mean).sample()

    # endregion

    @define_scope()
    def solve_w_mean(self,
                     memory_sample: tf.Tensor,
                     z: tf.Tensor) -> tf.Tensor:
        if self.use_addressing_matrix:
            raise NotImplementedError

        w_matrix = tf.matmul(memory_sample, memory_sample, transpose_b=True)
        w_rhs = tf.einsum("bmc,ebc->bme", memory_sample, z)
        l2_regularizer = tf.square(self._observational_noise_stddev) + tf.square(self._w_prior_stddev)
        w_mean = tf.matrix_solve_ls(matrix=w_matrix,
                                    rhs=w_rhs,
                                    l2_regularizer=l2_regularizer)
        w_mean = tf.transpose(w_mean, perm=[2, 0, 1], name="w_mean")
        return w_mean

    @staticmethod
    @define_scope()
    def z_mean_from_memory(w_sample: tf.Tensor, memory_sample: tf.Tensor, name="z_mean_from_memory"):
        return tf.einsum("ebm,bmc->ebc", w_sample, memory_sample, name=name)

    # region Kullback Leibler divergence
    @define_scope()
    def get_w_divergence(self, w_mean: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("w_kl_divergence"):
            return self.get_w_distribution(w_mean).kl_divergence(self.w_prior_distribution)

    @define_scope()
    def get_update_divergence(self,
                              memory_state: MemoryState,
                              w_sample: tf.Tensor,
                              z_mean: tf.Tensor,
                              ) -> tf.Tensor:
        with tf.name_scope("update_kl_divergence"):
            memory_mean, memory_row_covariance = memory_state
            predicted_z_mean = self.z_mean_from_memory(w_sample, memory_mean, name="predicted_z")
            prediction_delta = z_mean - predicted_z_mean
            prediction_delta_squared = tf.square(prediction_delta)

            cross_variance = tf.einsum("ebm,bmn->ebn", w_sample, memory_row_covariance,
                                       name="cross_variance_between_memory_and_w")
            variance_noise = tf.eye(tf.shape(cross_variance)[0], name="variance_noise_diagonal_matrix")
            tmp = tf.einsum("ebm,ebm->eb", cross_variance, w_sample)
            z_episode_covariance = tmp + variance_noise

            beta = tmp / z_episode_covariance
            beta2 = tf.expand_dims(beta / z_episode_covariance, axis=-1)
            code_size_float = tf.constant(self.code_size, dtype=backend.floatx())

            memory_kl_divergence = -0.5 * (
                    code_size_float * (beta + tf.log(1 - beta))
                    - tf.reduce_sum(beta2 * prediction_delta_squared,
                                    axis=-1)
            )
            return memory_kl_divergence

    # endregion

    def get_config(self) -> Dict[str, Any]:
        config = {
            "code_size": self.code_size,
            "memory_size": self.memory_size,
            "iteration_count": self.iteration_count,
            "w_prior_stddev": self.w_prior_stddev,
            "initial_w_stddev": self.initial_w_stddev,
            "observational_noise_stddev": self.observational_noise_stddev,
            "use_addressing_matrix": self.use_addressing_matrix
        }
        base_config = super(Memory, self).get_config()
        return {**base_config, **config}
