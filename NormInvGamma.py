# https://github.com/tensorflow/probability/issues/1647
# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionCoroutine
# https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution


# import jax
# import jax.numpy as jnp
# import tensorflow_probability.substrates.jax as tfp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root


def normal_inverse_gamma(mu, lam, alpha, beta):
    @tfd.JointDistributionCoroutine
    def model():
        sigma_2 = yield Root(tfd.InverseGamma(alpha, beta, name='sigma_2'))
        yield tfd.Normal(mu, sigma_2 / lam, name='x')

    return model


mu = 3
lam = 100
alpha = 2
beta = 0.5

E_sigma_2 = beta / (alpha - 1)
Var_mu = beta / (lam * (alpha - 1))

dist = normal_inverse_gamma(mu=mu,
                            lam=lam,
                            alpha=alpha,
                            beta=beta)
x = dist.sample(100)
px = dist.prob(x)
log_px = dist.log_prob(x)

print(f'x={x[1]}')
print(f'px={px}')
print(f'log_px={log_px}')
print(f'E_sigma_2={E_sigma_2}')
print(f'Var_mu={Var_mu}')

print(np.mean(x[1]))
print(np.std(x[1]))

"""
dist=normal_inverse_gamma(
    mu=3.0,    # real value
    lam=0.2,  # any positive
    alpha=1.0, # greater equal 1
    beta=1.0) # any positive

#x=dist.sample(3)
px=dist.prob(1.0)
log_px=dist.log_prob(1.0)

#print(x)
print(px)
print(log_px)


def model_fn():
  x = yield tfd.JointDistributionCoroutine.Root(
    tfd.Normal(0., tf.ones([3])))
  y = yield tfd.JointDistributionCoroutine.Root(
    tfd.Normal(0., 1.))
  z = yield tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)


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

"""

