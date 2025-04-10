# polars_rng

Random number generation in Polars.

## Usage

All random number generators are implmented as Polars expressions and can be used anywhere that expressions are allowed, including in lazy contexts.

```python
import polars as pl
import polars_rng as prng

# simulate data from a linear model
# y = x^2 + N(0, 3^2)
(
    pl.DataFrame(dict(x = [-3, 0, 2, 3, 4, 6, 11]))
    .with_columns(
        y = prng.normal(mu = pl.col("x").pow(2), sigma = 3)
    )
)
# shape: (7, 2)
# ┌─────┬────────────┐
# │ x   ┆ y          │
# │ --- ┆ ---        │
# │ i64 ┆ f64        │
# ╞═════╪════════════╡
# │ -3  ┆ 8.623695   │
# │ 0   ┆ -7.884036  │
# │ 2   ┆ 2.864818   │
# │ 3   ┆ 5.348727   │
# │ 4   ┆ 19.721068  │
# │ 6   ┆ 37.855376  │
# │ 11  ┆ 123.532992 │
# └─────┴────────────┘


# lazily simulate from binomial
lazy = (
    pl.LazyFrame()
    .select(n_coins = pl.repeat(10, 6))
    .with_columns(
        n_heads = prng.binomial(p = .5, n = "n_coins")
    )
)

lazy
# <LazyFrame at 0x1150E94F0>

lazy.collect()
# shape: (6, 2)
# ┌─────────┬─────────┐
# │ n_coins ┆ n_heads │
# │ ---     ┆ ---     │
# │ i32     ┆ f64     │
# ╞═════════╪═════════╡
# │ 10      ┆ 4.0     │
# │ 10      ┆ 7.0     │
# │ 10      ┆ 6.0     │
# │ 10      ┆ 3.0     │
# │ 10      ┆ 2.0     │
# │ 10      ┆ 5.0     │
# └─────────┴─────────┘

```

## Available Distributions

A full list of implmented distributions is available in the main namespace:

```python
from polars_rng import DISTRIBUTIONS

DISTRIBUTIONS
# ['normal',
# 'uniform',
# 'bernoulli',
# 'binomial',
# 'exponential',
# 'poisson',
# 'weibull',
# 'laplace',
# 'gamma',
# 'beta']
```
