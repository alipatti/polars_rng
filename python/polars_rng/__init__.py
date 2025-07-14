from typing import Any, Sequence, Union
import polars as pl

from polars_rng.helpers import into_expr, IntoExpr, sample
from polars.datatypes import DataTypeClass

# TODO: multivariate normal

# TODO: add mean and variance to docstrings

__all__ = [
    "normal",
    "uniform",
    "bernoulli",
    "binomial",
    "exponential",
    "poisson",
    "weibull",
    "laplace",
    "gamma",
    "beta",
]

DISTRIBUTIONS = __all__


def normal(mu: IntoExpr = 0, sigma: IntoExpr = 1) -> pl.Expr:
    """
    Samples from the normal (Gaussian) distribution with mean `mu` and standard deviation `sigma`.
    """
    return sample("standard_normal") * into_expr(sigma) + into_expr(mu)


def uniform(low: IntoExpr = 0, high: IntoExpr = 1) -> pl.Expr:
    """
    Samples from the uniform distribution on `[low, high]`.

    See: `https://en.wikipedia.org/wiki/Uniform_distribution`.
    """
    low, high = into_expr(low), into_expr(high)
    return sample("standard_uniform") * (high - low) + low


def uniform_integer(
    low: IntoExpr,
    high: IntoExpr,
    output_dtype: DataTypeClass = pl.UInt64,
    include_right=False,
) -> pl.Expr:
    """
    Samples uniformly from `{low, low + 1, ..., high - 1}` if `include_right` is `False` (default) or from `{low, low + 1, ..., high - 1}` if `include_right` is `True`.
    """

    return uniform(low, into_expr(high) + int(include_right)).cast(output_dtype)


def categorical(
    *,
    categories: Union[tuple[Any, ...], str, pl.Expr, None] = None,
    weights: Union[
        Union[tuple[int, ...], tuple[float, ...]], str, pl.Expr, None
    ] = None,
):
    if isinstance(categories, Sequence):
        categories = pl.repeat(
            list(categories), dtype=pl.List(type(categories[0])), n=pl.len()
        )
    elif isinstance(categories, (str, pl.Expr)):
        categories = into_expr(categories)

    if isinstance(weights, Sequence):
        weights = pl.repeat(list(weights), dtype=pl.List(float), n=pl.len())
    elif isinstance(weights, (str, pl.Expr)):
        weights = into_expr(weights)

    if categories is None:
        # take number of categories from probabilities
        if isinstance(weights, Sequence):
            categories = pl.int_ranges(0, len(weights))
        elif isinstance(weights, (str, pl.Expr)):
            categories = pl.int_ranges(0, into_expr(weights).list.len())
        else:
            raise ValueError("must provide at least one of categories or probabilities")

    if weights is None:
        sample_indices = uniform_integer(0, categories.list.len())
    else:
        cdf = weights.list.eval(pl.element().cum_sum() / pl.element().sum())
        sample_indices = cdf.list.eval(pl.element().search_sorted(uniform()))

    return categories.list.get(sample_indices).alias("sample")


def bernoulli(p: IntoExpr = 0.5) -> pl.Expr:
    """
    Samples from the Bernoulli distribution with success probability `p`.

    See: `https://en.wikipedia.org/wiki/Bernoulli_distribution`.
    """
    return sample("standard_uniform") < into_expr(p)


def binomial(p: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Samples from the binomial distribution with `n` trials and success probability `p`.

    See: `https://en.wikipedia.org/wiki/Binomial_distribution`.
    """
    p, n = into_expr(p, pl.Float64), into_expr(n, pl.UInt64)
    return sample("binomial", p, n)


def exponential(scale: IntoExpr = 1) -> pl.Expr:
    """
    Samples from the exponential distribution with scale parameter `scale`.
    Generated as `- scale * log(uniform(0, 1))`.

    See: `https://en.wikipedia.org/wiki/Exponential_distribution`.
    """
    return sample("standard_uniform").log().neg().mul(scale)


def poisson(rate: IntoExpr = 1) -> pl.Expr:
    """
    Samples from the Poisson distribution with rate parameter `rate`.

    See: `https://en.wikipedia.org/wiki/Poisson_distribution`.
    """
    rate = into_expr(rate, pl.Float64)
    return sample("poisson", rate)


def weibull(scale: IntoExpr = 1, shape: IntoExpr = 1):
    """
    Samples from the Weibull distribution with scale parameter `scale` and shape parameter `shape`.
    Generated as `exponential(scale) ** shape`.

    See: `https://en.wikipedia.org/wiki/Weibull_distribution`.
    """

    return exponential(scale).pow(shape)


def laplace(mu: IntoExpr = 0, b: IntoExpr = 1):
    """
    Samples from the Laplace distribution with mean `mu` and scale parameter `scale`.

    See: `https://en.wikipedia.org/wiki/Laplace_distribution`.
    """
    u = uniform(-1, 1)
    return u.sign() * into_expr(b) * (1 - u.abs()).log() + into_expr(mu)


def gamma(alpha: IntoExpr, theta: IntoExpr):
    """
    Samples from the gamma distribution with shape parameter `alpha` and scale parameter `theta`.

    See: `https://en.wikipedia.org/wiki/Gamma_distribution`.
    """
    alpha, theta = into_expr(alpha, pl.Float64), into_expr(theta, pl.Float64)
    return sample("gamma", alpha, theta)


def beta(alpha: IntoExpr, beta: IntoExpr):
    """
    Samples from the Weibull distribution with parameters `alpha` and `beta`.

    See: `https://en.wikipedia.org/wiki/Beta_distribution`.
    """
    alpha, beta = into_expr(alpha, pl.Float64), into_expr(beta, pl.Float64)
    return sample("beta", alpha, beta)


# aliases

gaussian = normal
