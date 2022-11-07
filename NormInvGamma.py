# https://github.com/tensorflow/probability/issues/1647
#import jax
#import jax.numpy as jnp
#import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability as tfp

tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root


def normal_inverse_gamma(mu, lam, alpha, beta):
  #@tfd.JointDistributionCoroutine
  def model():
    sigma_2 = yield Root(tfd.InverseGamma(alpha, beta, name='sigma_2'))
    yield tfd.Normal(mu, sigma_2 / lam, name='x')
  return model

dist = normal_inverse_gamma(1., 2., 3., 4.)
#draws = dist.sample(3, seed=jax.random.PRNGKey(0))
draws = dist.sample(3)
# StructTuple(
#   sigma_2=DeviceArray([2.2811928, 0.8736088, 1.801786 ], dtype=float32),
#   x=DeviceArray([1.8641728, 1.4615237, 1.8459291], dtype=float32)
# )
dist.log_prob(draws)
# DeviceArray([-2.9240332, -1.2213438, -2.364818 ], dtype=float32)