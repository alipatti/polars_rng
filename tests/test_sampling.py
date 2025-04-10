import polars as pl

import polars_rand as plr


def test_normal():

    pl.select(pl.int_range(1000)).select(
        plr.binomial(0.5, 10),
    )
