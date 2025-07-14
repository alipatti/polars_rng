from string import ascii_uppercase

import polars as pl

import polars_rng as prng

N = 100_000


def test_uniform_categorical():
    letters = pl.select(pl.repeat(None, N)).select(
        prng.categorical(categories=tuple(ascii_uppercase)),
    )

    assert letters.select(
        (pl.col("sample").unique_counts() / pl.len()).var() < 1 / N
    ).item()


def test_variable_length_categorical():
    lens = prng.uniform_integer(1, 26, include_right=True)

    pl.select(letters=pl.repeat(list(ascii_uppercase), N)).select(
        pl.col("letters").list.head(lens)
    )
