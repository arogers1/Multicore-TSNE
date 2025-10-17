"""Parallel t-SNE implementation using PySpark DataFrames and RDDs."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency during docs/tests
    from pyspark import SparkContext, StorageLevel
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.ml.linalg import DenseVector, SparseVector
except Exception:  # pragma: no cover - PySpark is optional for builds
    SparkContext = None  # type: ignore
    SparkSession = None  # type: ignore
    StorageLevel = None  # type: ignore
    DataFrame = None  # type: ignore
    DenseVector = None  # type: ignore
    SparseVector = None  # type: ignore


def _to_numpy_vector(value) -> np.ndarray:
    """Convert PySpark vectors and sequences to ``float64`` numpy arrays."""

    if isinstance(value, np.ndarray):
        arr = value
    elif SparseVector is not None and isinstance(value, SparseVector):
        arr = value.toArray()
    elif DenseVector is not None and isinstance(value, DenseVector):
        arr = value.values
    else:
        arr = np.asarray(value)
    return np.asarray(arr, dtype=np.float64)


@dataclass(frozen=True)
class _StateTuple:
    """Container storing the optimisation state for a single sample."""

    embedding: np.ndarray
    gains: np.ndarray
    updates: np.ndarray


class SparkTSNE:
    """Distributed t-SNE solver executed on top of Spark."""

    def __init__(
        self,
        spark: "SparkSession | SparkContext",
        *,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iters: int = 250,
        momentum: float = 0.5,
        final_momentum: float = 0.8,
        min_gain: float = 0.01,
        random_state: Optional[numbers.Integral] = None,
        num_partitions: Optional[int] = None,
        use_spartann: bool = False,
        verbose: bool = True,
        features_col: str = "features",
    ) -> None:
        if SparkContext is None:
            raise RuntimeError(
                "pyspark is not available. Install pyspark to use SparkTSNE."
            )

        if isinstance(spark, SparkSession):
            self._spark = spark
            self._sc = spark.sparkContext
        elif isinstance(spark, SparkContext):
            self._spark = None
            self._sc = spark
        else:
            raise TypeError(
                "spark must be a SparkSession or SparkContext, got %r" % (spark,)
            )

        if n_iter < 250:
            raise ValueError("n_iter must be at least 250 to include early exaggeration")
        if perplexity <= 0:
            raise ValueError("perplexity must be greater than 0")
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")

        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.early_exaggeration = float(early_exaggeration)
        self.early_exaggeration_iters = int(early_exaggeration_iters)
        self.momentum = float(momentum)
        self.final_momentum = float(final_momentum)
        self.min_gain = float(min_gain)
        self.random_state = random_state
        self.num_partitions = num_partitions
        self.use_spartann = bool(use_spartann)
        self.verbose = bool(verbose)
        self.features_col = features_col

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit_transform(self, X) -> np.ndarray:
        """Run t-SNE on the provided data and return the embedding."""

        indexed_rdd, n_samples, n_features = self._prepare_input(X)
        if self.perplexity * 3 > n_samples:
            raise ValueError(
                "perplexity is too large for the number of samples; expected at least"
                f" {int(self.perplexity * 3)} samples but got {n_samples}"
            )

        self._log(
            "Starting SparkTSNE on %d samples with %d features" %
            (n_samples, n_features)
        )

        distance_pairs = self._compute_pairwise_distances(indexed_rdd).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        probabilities = self._compute_joint_probabilities(distance_pairs, n_samples).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        distance_pairs.unpersist()

        state = self._initialise_state(indexed_rdd, n_samples)
        indexed_rdd.unpersist()

        final_state = self._optimise(state, probabilities, n_samples)
        probabilities.unpersist()

        embedding = (
            final_state
            .mapValues(lambda value: value.embedding)
            .sortByKey()
            .values()
            .collect()
        )
        final_state.unpersist()
        return np.vstack(embedding)

    # ------------------------------------------------------------------
    # helpers for preparing the data
    # ------------------------------------------------------------------
    def _prepare_input(self, data) -> Tuple:
        if isinstance(data, np.ndarray):
            features = np.asarray(data, dtype=np.float64)
            if features.ndim != 2:
                raise ValueError("X must be a 2-D array")
            rdd = self._sc.parallelize(features.tolist(), self.num_partitions)
        elif DataFrame is not None and isinstance(data, DataFrame):
            if self.features_col not in data.columns:
                raise ValueError(
                    f"features column '{self.features_col}' not found in DataFrame"
                )
            rdd = data.select(self.features_col).rdd.map(lambda row: row[0])
        else:
            try:
                rdd = self._sc.parallelize(list(data), self.num_partitions)
            except TypeError as exc:  # pragma: no cover - defensive programming
                raise TypeError(
                    "X must be a numpy array, Spark DataFrame, or iterable"
                ) from exc

        indexed = (
            rdd
            .map(_to_numpy_vector)
            .zipWithIndex()
            .map(lambda pair: (int(pair[1]), pair[0]))
        )
        if self.num_partitions is not None:
            indexed = indexed.repartition(self.num_partitions)
        indexed = indexed.cache()

        n_samples = indexed.count()
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")
        first_vector = indexed.take(1)[0][1]
        n_features = int(first_vector.shape[0])
        return indexed, n_samples, n_features

    # ------------------------------------------------------------------
    # helpers for distance / probability computation
    # ------------------------------------------------------------------
    def _compute_pairwise_distances(self, indexed_rdd):
        def _distances(pair):
            (i, xi), (j, xj) = pair
            if i > j:
                return []
            if i == j:
                return [((i, i), 0.0)]
            diff = xi - xj
            dist = float(np.dot(diff, diff))
            return [((i, j), dist), ((j, i), dist)]

        return indexed_rdd.cartesian(indexed_rdd).flatMap(_distances)

    def _compute_joint_probabilities(self, distance_pairs, n_samples: int):
        log_perplexity = math.log(self.perplexity, 2)
        tiny = np.finfo(np.float64).tiny

        grouped = distance_pairs.map(
            lambda kv: (kv[0][0], (kv[0][1], kv[1]))
        ).groupByKey()

        def _conditional(entry):
            idx, values = entry
            row = np.full(n_samples, np.inf, dtype=np.float64)
            for j, dist in values:
                row[j] = dist
            row[idx] = 0.0
            beta = 1.0
            beta_min = None
            beta_max = None
            target = log_perplexity
            entropy = 0.0
            prob = np.zeros(n_samples, dtype=np.float64)
            for _ in range(50):
                exp_row = np.exp(-(row * beta))
                exp_row[idx] = 0.0
                sum_exp = float(np.sum(exp_row))
                if sum_exp == 0.0:
                    sum_exp = tiny
                prob = exp_row / sum_exp
                entropy = -np.sum(prob * np.log2(np.maximum(prob, tiny)))
                diff = entropy - target
                if abs(diff) < 1e-5:
                    break
                if diff > 0:
                    beta_min = beta
                    beta = beta * 2 if beta_max is None else (beta + beta_max) / 2
                else:
                    beta_max = beta
                    beta = beta / 2 if beta_min is None else (beta + beta_min) / 2
            return idx, prob

        conditionals = grouped.map(_conditional).persist(StorageLevel.MEMORY_AND_DISK)

        def _symmetrise(entry):
            idx, row = entry
            for j, value in enumerate(row):
                if j == idx:
                    continue
                yield (min(idx, j), max(idx, j)), float(value)

        summed = conditionals.flatMap(_symmetrise).reduceByKey(lambda a, b: a + b)

        def _expand(entry):
            (i, j), value = entry
            prob = max(value / (2.0 * n_samples), tiny)
            return [((i, j), prob), ((j, i), prob)]

        expanded = summed.flatMap(_expand)
        diag = self._sc.parallelize(
            [((i, i), tiny) for i in range(n_samples)],
            self.num_partitions or self._sc.defaultParallelism,
        )
        all_pairs = expanded.union(diag).persist(StorageLevel.MEMORY_AND_DISK)
        total = all_pairs.map(lambda kv: kv[1]).sum()
        norm = max(total, tiny)
        joint = all_pairs.mapValues(lambda value: max(value / norm, tiny))
        all_pairs.unpersist()
        conditionals.unpersist()
        return joint

    # ------------------------------------------------------------------
    # optimisation loop
    # ------------------------------------------------------------------
    def _initialise_state(self, indexed_rdd, n_samples: int):
        base_seed = None if self.random_state is None else int(self.random_state)

        def _init(entry: Tuple[int, np.ndarray]):
            idx, _ = entry
            seed = None if base_seed is None else base_seed + idx
            rng = np.random.default_rng(seed)
            emb = 1e-4 * rng.standard_normal(self.n_components)
            gains = np.ones(self.n_components, dtype=np.float64)
            updates = np.zeros(self.n_components, dtype=np.float64)
            return idx, _StateTuple(embedding=emb, gains=gains, updates=updates)

        state = indexed_rdd.map(_init)
        return state.persist(StorageLevel.MEMORY_AND_DISK)

    def _optimise(self, state_rdd, probabilities, n_samples: int):
        degrees_of_freedom = max(self.n_components - 1.0, 1.0)
        exaggeration = self.early_exaggeration
        momentum = self.momentum

        state = state_rdd
        for iteration in range(self.n_iter):
            if iteration == self.early_exaggeration_iters:
                exaggeration = 1.0
                momentum = self.final_momentum

            gradients, kl = self._gradient_step(state, probabilities, degrees_of_freedom, exaggeration)
            state = self._apply_gradients(state, gradients, momentum)
            state = self._zero_mean(state, n_samples)

            if self.verbose and (iteration % 50 == 0 or iteration + 1 == self.n_iter):
                self._log(
                    "Iteration %d/%d, KL divergence %.6f" %
                    (iteration + 1, self.n_iter, kl)
                )

        return state

    def _gradient_step(self, state_rdd, probabilities, degrees_of_freedom: float, exaggeration: float):
        tiny = np.finfo(np.float64).tiny

        embedding_rdd = state_rdd.mapValues(lambda value: value.embedding).persist(
            StorageLevel.MEMORY_AND_DISK
        )

        def _pairwise(pair):
            (i, yi), (j, yj) = pair
            if i == j:
                return []
            diff = yi - yj
            dist = float(np.dot(diff, diff) / degrees_of_freedom)
            num = 1.0 / (1.0 + dist)
            return [((i, j), (num, diff))]

        numerators = embedding_rdd.cartesian(embedding_rdd).flatMap(_pairwise)
        numerators.persist(StorageLevel.MEMORY_AND_DISK)

        denom = numerators.map(lambda kv: kv[1][0]).sum()
        denom = max(denom, tiny)

        joined = numerators.join(probabilities)
        denom_bc = self._sc.broadcast(denom)

        def _contribution(entry):
            (i, j), ((num, diff), prob) = entry
            q_val = num / denom_bc.value
            weight = (prob * exaggeration - q_val) * num
            grad = 4.0 * weight * diff
            return i, grad

        gradients = joined.map(_contribution).reduceByKey(lambda a, b: a + b)

        def _kl_term(entry):
            (_, ((num, _), prob)) = entry
            q_val = num / denom_bc.value
            return prob * math.log(max(prob, tiny) / max(q_val, tiny))

        kl_divergence = float(joined.map(_kl_term).sum())

        denom_bc.unpersist(blocking=False)
        numerators.unpersist()
        embedding_rdd.unpersist()

        return gradients, kl_divergence

    def _apply_gradients(self, state_rdd, gradients, momentum: float):
        def _update(entry):
            idx, (state, grad) = entry
            embedding = state.embedding
            gains = state.gains
            updates = state.updates
            gradient = grad if grad is not None else np.zeros_like(embedding)
            sign_match = np.sign(gradient) == np.sign(updates)
            gains = np.where(sign_match, gains * 0.8, gains + 0.2)
            gains = np.maximum(gains, self.min_gain)
            updates = momentum * updates - self.learning_rate * (gains * gradient)
            embedding = embedding + updates
            return idx, _StateTuple(embedding=embedding, gains=gains, updates=updates)

        joined = state_rdd.leftOuterJoin(gradients)
        updated = joined.map(_update).persist(StorageLevel.MEMORY_AND_DISK)
        state_rdd.unpersist()
        return updated

    def _zero_mean(self, state_rdd, n_samples: int):
        def _add(a, b):
            return a + b

        sum_embedding = state_rdd.map(lambda kv: kv[1].embedding).reduce(_add)
        mean = sum_embedding / float(n_samples)
        mean_bc = self._sc.broadcast(mean)

        def _center(entry):
            idx, value = entry
            centered = value.embedding - mean_bc.value
            return idx, _StateTuple(embedding=centered, gains=value.gains, updates=value.updates)

        centered = state_rdd.map(_center).persist(StorageLevel.MEMORY_AND_DISK)
        state_rdd.unpersist()
        mean_bc.unpersist(blocking=False)
        return centered

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.verbose:
            print("[SparkTSNE] %s" % message)

    # ------------------------------------------------------------------
    # optional integrations
    # ------------------------------------------------------------------
    def _attempt_spartann_neighbors(self, features: np.ndarray) -> Optional[np.ndarray]:
        if self._spark is None:
            self._log(
                "SparkSession is not available; com.nosto.spartann integration "
                "requires a SparkSession for DataFrame interoperability."
            )
            return None

        try:
            _ = self._sc._jvm.com.nosto.spartann
        except AttributeError:
            self._log(
                "com.nosto.spartann was not detected on the Spark classpath; "
                "falling back to the exact distance computation."
            )
            return None

        self._log(
            "com.nosto.spartann detected. Automatic integration is not shipped "
            "by default because different installations expose different entry "
            "points. Override `_attempt_spartann_neighbors` if you want to "
            "delegate neighbour search to SpartANN."
        )
        return None


__all__ = ["SparkTSNE"]

