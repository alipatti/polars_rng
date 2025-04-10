use polars_core::chunked_array::ChunkedArray;
use polars_core::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::prelude::Distribution as _;
use rand::rngs::ThreadRng;
use serde::Deserialize;
use statrs::distribution::{Beta, Binomial, Gamma, Normal, Poisson, Uniform};

#[derive(Deserialize)]
struct SampleKwargs {
    distribution: Distribution,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Distribution {
    StandardNormal,
    StandardUniform,
    Binomial,
    Gamma,
    Poisson,
    Beta,
}

// TODO: refactor this function (tons of repetition, but i also wrote in one evening so i don't
// feel too bad)
// PERF: paralellize this jawn
#[polars_expr(output_type=Float64)]
fn sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let n = inputs[0].len();
    let mut rng = ThreadRng::default();

    let arr = match kwargs.distribution {
        // can use downstream for normal (arbitrary), log normal, chi square (small n)
        Distribution::StandardNormal => {
            let dist = Normal::new(0.0, 1.0).unwrap(); // N(0, 1) will never fail
            let iter = dist.sample_iter(&mut rng).take(n);
            ChunkedArray::<Float64Type>::from_iter_values("sample".into(), iter)
        }
        // can use downstream for bern, exp, etc.
        Distribution::StandardUniform => {
            let dist = Uniform::new(0.0, 1.0).unwrap(); // U(0, 1) will never fail
            let iter = dist.sample_iter(&mut rng).take(n);
            ChunkedArray::<Float64Type>::from_iter_values("sample".into(), iter)
        }
        Distribution::Binomial => {
            let ps = inputs[0].f64()?;
            let ns = inputs[1].u64()?;

            let iter = ps.iter().zip(ns.iter()).map(|(p, n)| {
                if let (Some(p), Some(n)) = (p, n) {
                    Binomial::new(p, n).ok().map(|dist| dist.sample(&mut rng))
                } else {
                    None
                }
            });

            ChunkedArray::<Float64Type>::from_iter_options(
                "sample".into(),
                iter,
            )
        }
        Distribution::Gamma => {
            let shapes = inputs[0].f64()?;
            let scales = inputs[1].f64()?;

            let iter =
                shapes.iter().zip(scales.iter()).map(|(alpha, theta)| {
                    if let (Some(alpha), Some(theta)) = (alpha, theta) {
                        Gamma::new(alpha, 1.0 / theta)
                            .ok()
                            .map(|dist| dist.sample(&mut rng))
                    } else {
                        None
                    }
                });

            ChunkedArray::<Float64Type>::from_iter_options(
                "sample".into(),
                iter,
            )
        }
        Distribution::Beta => {
            let alpha = inputs[0].f64()?;
            let beta = inputs[1].f64()?;

            let iter = alpha.iter().zip(beta.iter()).map(|(alpha, beta)| {
                if let (Some(alpha), Some(beta)) = (alpha, beta) {
                    Beta::new(alpha, beta)
                        .ok()
                        .map(|dist| dist.sample(&mut rng))
                } else {
                    None
                }
            });

            ChunkedArray::<Float64Type>::from_iter_options(
                "sample".into(),
                iter,
            )
        }
        Distribution::Poisson => {
            let rates = inputs[0].f64()?;

            let iter = rates.iter().map(|lambda| {
                if let Some(lambda) = lambda {
                    Poisson::new(lambda).ok().map(|dist| dist.sample(&mut rng))
                } else {
                    None
                }
            });

            ChunkedArray::<Float64Type>::from_iter_options(
                "sample".into(),
                iter,
            )
        }
    };

    Ok(arr.into_series())
}
