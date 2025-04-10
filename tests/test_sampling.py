import pytest
from math import sqrt
from typing import Any, Callable, Union
import polars as pl

import polars_rng as plr


# TODO: set a seed

Params = tuple[Union[float, int], ...]
s = pl.col("sample")


TEST_CASES = [
    # normal
    (plr.normal, (0, 1)),
    (plr.normal, (-1928, 113)),
    (plr.normal, (0, 0.01)),
    # uniform
    (plr.uniform, (0, 1)),
    (plr.uniform, (0, 0.01)),
    (plr.uniform, (-1000, 1000)),
    # exponential
    (plr.exponential, (1,)),
    (plr.exponential, (140,)),
    (plr.exponential, (0.014,)),
    # binomial
    (plr.binomial, (0.5, 20)),
    (plr.binomial, (0.05, 200)),
    # bern
    (plr.bernoulli, (0.5,)),
    (plr.bernoulli, (0.01,)),
    # pois
    (plr.poisson, (1,)),
    (plr.poisson, (0.016,)),
    (plr.poisson, (150,)),
    # laplace
    (plr.laplace, (0, 1)),
    (plr.laplace, (-10, 10)),
    (plr.laplace, (100, 15)),
    # gamma
    (plr.gamma, (1, 1)),
    (plr.gamma, (0.3, 10)),
    (plr.gamma, (10, 7)),
    # beta
    (plr.beta, (1, 1)),
    (plr.beta, (10, 10)),
    (plr.beta, (0.5, 0.5)),  # jeff's for binomial
]


@pytest.mark.parametrize("distribution, parameters", TEST_CASES)
def test_distribution(
    distribution: Callable[..., pl.Expr],
    parameters: Params,
    *,
    # test params
    n=10_000,
    n_std_devs=5,
):
    pop_mean, pop_var = get_mean_and_var(distribution, parameters)

    sample = (
        pl.select(pl.int_range(n))
        .with_columns(mean=pop_mean, std_dev=sqrt(pop_var))
        .with_columns(distribution(*parameters))
    )

    # https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Distribution_of_the_sample_mean
    sample_mean = sample.select(s.mean())[0, 0]
    assert abs(pop_mean - sample_mean) < n_std_devs * sqrt(pop_var / n), (
        f"{pop_mean=:.3f}, {sample_mean=:.3f}"
    )

    # https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance

    sample_var = sample.select(s.var())[0, 0]
    assert abs(pop_var - sample_var) < n_std_devs * 2 * sqrt(
        2 * pop_var**2 / (n - 1)
    ), f"{pop_var=:.3f}, {sample_var=:.3f}"


def get_mean_and_var(distribution: Any, params: Params) -> tuple[float, float]:
    if distribution is plr.normal:
        return params[0], params[1] ** 2

    if distribution is plr.uniform:
        return (
            (params[0] + params[1]) / 2,
            1 / 12 * (params[1] - params[0]) ** 2,
        )

    if distribution is plr.exponential:
        return params[0], params[0] ** 2

    if distribution is plr.bernoulli:
        return params[0], params[0] * (1 - params[0])

    if distribution is plr.binomial:
        return params[0] * params[1], params[0] * (1 - params[0]) * params[1]

    if distribution is plr.poisson:
        return params[0], params[0]

    if distribution is plr.laplace:
        return params[0], 2 * params[1] ** 2

    if distribution is plr.laplace:
        return params[0], 2 * params[1] ** 2

    if distribution is plr.beta:
        a, b = params
        return (
            a / (a + b),
            (a * b) / ((a + b) ** 2 * (a + b + 1)),
        )

    if distribution is plr.gamma:
        alpha, theta = params
        return alpha * theta, alpha * theta**2

    else:
        raise ValueError(f"invalid distribution: {distribution}")
