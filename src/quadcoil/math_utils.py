import jax.numpy as jnp

sin_or_cos = lambda x, mode: jnp.where(mode==1, jnp.sin(x), jnp.cos(x))