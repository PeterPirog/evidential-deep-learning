# NORMAL DISTRIBUTION
import tensorflow_probability as tfp
tfd =tfp.distributions

dist=tfd.Normal(loc=[3],scale=1.5)

x=dist.sample(3)
px=dist.prob(x)
log_px=dist.log_prob(x)
mu=dist.mean()
std=dist.stddev()
print(x)
print(px)
print(log_px)
print(mu)
print(std)

# SINH-ARCSINH DISTRIBUTION
# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/SinhArcsinh
# https://www.r-bloggers.com/2019/04/the-sinh-arcsinh-normal-distribution/
# https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1740-9713.2019.01245.x
print('SINH-ARCSINH DISTRIBUTION')
dist=tfd.SinhArcsinh(
    loc=3.0,
    scale=1,
    skewness=None,
    tailweight=None)
x=dist.sample(3)
px=dist.prob(x)
log_px=dist.log_prob(x)
mu=dist.mean()
std=dist.stddev()
print(x)
print(px)
print(log_px)
print(mu)
print(std)