import polars as pl

from polars_rand.helpers import into_expr, IntoExpr, sample

# TODO: write docstrings


def normal(mean: IntoExpr = 0, std_dev: IntoExpr = 1) -> pl.Expr:
    return sample("standard_normal") * into_expr(std_dev) + into_expr(mean)


def uniform(low: IntoExpr = 0, high: IntoExpr = 1) -> pl.Expr:
    low, high = into_expr(low), into_expr(high)
    return sample("standard_uniform") * (high - low) + low


def bernoulli(p: IntoExpr = 0.5) -> pl.Expr:
    return sample("standard_uniform") < into_expr(p)


def binomial(p: IntoExpr, n: IntoExpr) -> pl.Expr:
    p, n = into_expr(p, pl.Float64), into_expr(n, pl.UInt64)
    return sample("binomial", p, n)


def exponential(scale: IntoExpr = 1) -> pl.Expr:
    return sample("standard_uniform").log().neg().mul(scale)


def poisson(rate: IntoExpr = 1) -> pl.Expr:
    rate = into_expr(rate, pl.Float64)
    return sample("poisson", rate)


def weibull(scale: IntoExpr = 1, shape: IntoExpr = 1):
    return exponential(scale).pow(shape)


def laplace(mean: IntoExpr = 0, scale: IntoExpr = 1):
    u = uniform(-1, 1)
    return u.sign() * into_expr(scale) * (1 - u.abs()).log() + into_expr(mean)


def gamma(shape: IntoExpr, scale: IntoExpr):
    shape, scale = into_expr(shape, pl.Float64), into_expr(scale, pl.Float64)
    return sample("gamma", shape, scale)


def beta(alpha: IntoExpr, beta: IntoExpr):
    alpha, beta = into_expr(alpha, pl.Float64), into_expr(beta, pl.Float64)
    return sample("beta", alpha, beta)
