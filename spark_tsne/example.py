"""Minimal usage example for :mod:`spark_tsne`.

The script demonstrates how to launch :class:`SparkTSNE` on a synthetic dataset.
It can be executed with ``spark-submit`` and will print the resulting embedding
coordinates to stdout.  The example deliberately keeps the dimensionality small
so that it can run on a development machine.
"""
from __future__ import annotations

import argparse

import numpy as np
from pyspark.sql import SparkSession

from .pyspark_tsne import SparkTSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SparkTSNE on synthetic data")
    parser.add_argument("--samples", type=int, default=512, help="Number of synthetic samples")
    parser.add_argument("--features", type=int, default=32, help="Number of features per sample")
    parser.add_argument("--iterations", type=int, default=750, help="Number of optimisation iterations")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("SparkTSNEExample").getOrCreate()

    rng = np.random.default_rng(args.seed)
    base = rng.standard_normal(size=(args.samples // 2, args.features))
    offset = rng.standard_normal(size=(args.features,)) * 5.0
    data = np.vstack([base, base + offset])

    solver = SparkTSNE(
        spark,
        n_iter=max(args.iterations, 250),
        perplexity=30.0,
        random_state=args.seed,
        verbose=True,
    )
    embedding = solver.fit_transform(data)

    print("Embedding shape:", embedding.shape)
    print(embedding)
    spark.stop()


if __name__ == "__main__":
    main()
