"""Parallel t-SNE implementation using PySpark.

This module provides :class:`SparkTSNE`, a close sibling of the original
:class:`MulticoreTSNE` implementation bundled with this repository.  Instead of
relying on shared-memory parallelism through OpenMP the heavy computations are
expressed as Spark jobs that can run on a cluster.  The implementation keeps the
original Barnes-Hut free ("exact") optimisation strategy to make the code a bit
simpler to reason about when running on a distributed system.  Nevertheless, the
per-point computations such as the probability estimation and gradient updates
are executed in parallel on the Spark executors.

The implementation is intentionally opinionated: it expects a ``SparkSession``
(or a ``SparkContext``) to be provided and works with dense ``numpy`` arrays.
The array is broadcast to the executors so that each executor can locally
compute the distances from its assigned subset of points to the entire
collection.  While this still has the same theoretical complexity as the exact
algorithm, it allows the workload to be evenly distributed across machines in a
cluster and gives an easy integration point for approximate neighbour libraries
such as ``com.nosto.spartann`` if they are available on the classpath.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
except Exception as exc:  # pragma: no cover - import is optional during doc builds
    SparkContext = None  # type: ignore
    SparkSession = None  # type: ignore


@dataclass(frozen=True)
class _TSNEState:
    """State container broadcast to workers during optimisation."""

    embedding: np.ndarray
    probabilities: np.ndarray
    degrees_of_freedom: float
    denominator: float


class SparkTSNE:
    """Distributed t-SNE solver executed on top of Spark.

    Parameters
    ----------
    spark : :class:`SparkSession` or :class:`SparkContext`
        The active Spark entry point that should be used to execute the
        distributed jobs.  If a ``SparkSession`` is supplied the underlying
        ``SparkContext`` is used automatically.
    n_components : int, optional
        Dimensionality of the embedding space.  Only values of ``2`` or ``3``
        are commonly used, but any positive integer is supported.
    perplexity : float, optional
        Desired perplexity for the conditional probability distribution.  The
        implementation mirrors the behaviour of scikit-learn and the original
        Multicore-TSNE project.
    learning_rate : float, optional
        Step size used during gradient descent.
    n_iter : int, optional
        Number of optimisation iterations.  At least ``250`` iterations are
        required because of the early exaggeration stage.
    early_exaggeration : float, optional
        Coefficient applied to the input similarities during the initial phase
        of the optimisation.
    early_exaggeration_iters : int, optional
        Number of iterations to keep early exaggeration enabled.
    momentum : float, optional
        Initial momentum used while updating the embedding.
    final_momentum : float, optional
        Momentum used once the early exaggeration stage has finished.
    min_gain : float, optional
        Floor for the adaptive gains, inherited from the reference
        implementation.
    random_state : int or :class:`numpy.random.Generator`, optional
        Random seed / generator controlling the embedding initialisation.
    num_partitions : int, optional
        Controls the number of Spark partitions used for computations.  By
        default the current ``SparkContext.defaultParallelism`` value is used.
    use_spartann : bool, optional
        When ``True`` the solver will try to rely on the ``com.nosto.spartann``
        package to obtain approximate nearest neighbours.  If the package is not
        available the code gracefully falls back to the exact computation.
    verbose : bool, optional
        Enable progress logging.
    """

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
    ) -> None:
        if SparkContext is None:
            raise RuntimeError(
                "pyspark is not available. Install pyspark to use SparkTSNE.")

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

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Run t-SNE on the provided array and return the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix containing the samples that need to be embedded.

        Returns
        -------
        ndarray of shape (n_samples, ``n_components``)
            Low-dimensional embedding.
        """

        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("X must be a 2-D array")

        n_samples, n_features = features.shape
        if self.perplexity * 3 > n_samples:
            raise ValueError(
                "perplexity is too large for the number of samples; expected at least"
                f" {int(self.perplexity * 3)} samples but got {n_samples}"
            )

        self._log(
            "Starting SparkTSNE on %d samples with %d features" %
            (n_samples, n_features)
        )

        indices_rdd = self._sc.parallelize(
            range(n_samples),
            self.num_partitions or self._sc.defaultParallelism,
        ).cache()

        pairwise_distances = self._compute_pairwise_distances(indices_rdd, features)
        probabilities = self._compute_joint_probabilities(indices_rdd, pairwise_distances)
        embedding = self._initialise_embedding(n_samples, self.n_components)

        state = self._optimise(indices_rdd, probabilities, embedding)
        indices_rdd.unpersist()
        return state.embedding

    # ------------------------------------------------------------------
    # helpers for distance / probability computation
    # ------------------------------------------------------------------
    def _compute_pairwise_distances(
        self,
        indices_rdd,
        features: np.ndarray,
    ) -> np.ndarray:
        """Compute the squared Euclidean distance matrix in parallel."""

        if self.use_spartann:
            approximation = self._attempt_spartann_neighbors(features)
            if approximation is not None:
                return approximation

        features_bc = self._sc.broadcast(features)

        def _distances(partition: Iterable[int]) -> Iterator[Tuple[int, np.ndarray]]:
            data = features_bc.value
            for idx in partition:
                diff = data[idx] - data
                # squared Euclidean distances
                dists = np.sum(diff * diff, axis=1)
                dists[idx] = 0.0
                yield idx, dists

        rows = dict(indices_rdd.mapPartitions(_distances).collect())
        features_bc.unpersist(blocking=False)

        ordered = np.empty((features.shape[0], features.shape[0]), dtype=np.float64)
        for idx in range(features.shape[0]):
            ordered[idx] = rows[idx]
        return ordered

    def _compute_joint_probabilities(
        self,
        indices_rdd,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Estimate joint probabilities for the input data."""

        log_perplexity = math.log(self.perplexity, 2)
        distances_bc = self._sc.broadcast(distances)

        def _probabilities(partition: Iterable[int]) -> Iterator[Tuple[int, np.ndarray]]:
            dist = distances_bc.value
            for idx in partition:
                row = dist[idx]
                beta = 1.0
                beta_min = None
                beta_max = None
                entropy = 0.0
                target = log_perplexity
                for _ in range(50):
                    p_row = np.exp(-row * beta)
                    p_row[idx] = 0.0
                    sum_p = p_row.sum()
                    if sum_p == 0.0:
                        sum_p = np.finfo(np.float64).tiny
                    p_row /= sum_p
                    entropy = -np.sum(p_row * np.log2(np.maximum(p_row, np.finfo(np.float64).tiny)))
                    perp_diff = entropy - target
                    if abs(perp_diff) < 1e-5:
                        break
                    if perp_diff > 0:
                        beta_min = beta
                        beta = beta * 2 if beta_max is None else (beta + beta_max) / 2
                    else:
                        beta_max = beta
                        beta = beta / 2 if beta_min is None else (beta + beta_min) / 2
                yield idx, p_row

        rows = dict(indices_rdd.mapPartitions(_probabilities).collect())
        distances_bc.unpersist(blocking=False)

        ordered = np.empty_like(distances)
        for idx in range(distances.shape[0]):
            ordered[idx] = rows[idx]

        P = ordered + ordered.T
        P /= np.maximum(np.sum(P), np.finfo(np.float64).tiny)
        P = np.maximum(P, np.finfo(np.float64).tiny)
        return P

    # ------------------------------------------------------------------
    # optimisation loop
    # ------------------------------------------------------------------
    def _optimise(
        self,
        indices_rdd,
        probabilities: np.ndarray,
        embedding: np.ndarray,
    ) -> _TSNEState:
        n_samples = probabilities.shape[0]
        gains = np.ones_like(embedding)
        updates = np.zeros_like(embedding)
        P = probabilities.copy()
        P *= self.early_exaggeration

        degrees_of_freedom = max(self.n_components - 1.0, 1.0)
        momentum = self.momentum

        for iteration in range(self.n_iter):
            if iteration == self.early_exaggeration_iters:
                P /= self.early_exaggeration
                momentum = self.final_momentum

            state = self._gradient_step(indices_rdd, embedding, P, degrees_of_freedom)
            grads = state.embedding  # gradient stored in the embedding field temporarily

            sign_match = np.sign(grads) == np.sign(updates)
            gains = (gains + 0.2) * (~sign_match) + (gains * 0.8) * sign_match
            gains[gains < self.min_gain] = self.min_gain

            updates = momentum * updates - self.learning_rate * (gains * grads)
            embedding += updates
            embedding -= np.mean(embedding, axis=0)

            if self.verbose and (iteration % 50 == 0 or iteration + 1 == self.n_iter):
                kl_divergence = self._kl_divergence(embedding, P, degrees_of_freedom)
                self._log(
                    "Iteration %d/%d, KL divergence %.6f" %
                    (iteration + 1, self.n_iter, kl_divergence)
                )

        return _TSNEState(embedding=embedding, probabilities=P, degrees_of_freedom=degrees_of_freedom, denominator=0.0)

    def _gradient_step(
        self,
        indices_rdd,
        embedding: np.ndarray,
        probabilities: np.ndarray,
        degrees_of_freedom: float,
    ) -> _TSNEState:
        embedding_bc = self._sc.broadcast(embedding)
        probabilities_bc = self._sc.broadcast(probabilities)

        def _denominator(partition: Iterable[int]) -> Iterator[float]:
            Y = embedding_bc.value
            for idx in partition:
                diff = Y[idx] - Y
                num = 1.0 / (1.0 + np.sum(diff * diff, axis=1) / degrees_of_freedom)
                num[idx] = 0.0
                yield float(np.sum(num))

        denom = indices_rdd.mapPartitions(_denominator).sum()
        if denom <= 0.0:
            denom = np.finfo(np.float64).tiny
        denom_bc = self._sc.broadcast(denom)

        def _gradient(partition: Iterable[int]) -> Iterator[Tuple[int, np.ndarray]]:
            Y = embedding_bc.value
            P = probabilities_bc.value
            denom_val = denom_bc.value
            for idx in partition:
                diff = Y[idx] - Y
                num = 1.0 / (1.0 + np.sum(diff * diff, axis=1) / degrees_of_freedom)
                num[idx] = 0.0
                q_row = num / denom_val
                pq = P[idx] - q_row
                grad = 4.0 * np.sum((pq[:, None] * num[:, None]) * diff, axis=0)
                yield idx, grad

        grads = dict(indices_rdd.mapPartitions(_gradient).collect())

        embedding_bc.unpersist(blocking=False)
        probabilities_bc.unpersist(blocking=False)
        denom_bc.unpersist(blocking=False)

        ordered = np.empty_like(embedding)
        for idx in range(embedding.shape[0]):
            ordered[idx] = grads[idx]
        return _TSNEState(embedding=ordered, probabilities=probabilities, degrees_of_freedom=degrees_of_freedom, denominator=denom)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def _initialise_embedding(self, n_samples: int, n_components: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return 1e-4 * rng.standard_normal(size=(n_samples, n_components))

    def _kl_divergence(
        self,
        embedding: np.ndarray,
        probabilities: np.ndarray,
        degrees_of_freedom: float,
    ) -> float:
        diff = embedding[:, None, :] - embedding[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2) / degrees_of_freedom
        num = 1.0 / (1.0 + dist_sq)
        np.fill_diagonal(num, 0.0)
        denom = np.sum(num)
        q = num / np.maximum(denom, np.finfo(np.float64).tiny)
        return float(np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny) / np.maximum(q, np.finfo(np.float64).tiny))))

    def _log(self, message: str) -> None:
        if self.verbose:
            print("[SparkTSNE] %s" % message)

    # ------------------------------------------------------------------
    # optional integrations
    # ------------------------------------------------------------------
    def _attempt_spartann_neighbors(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Try to compute neighbours using com.nosto.spartann if available.

        The SpartANN project exposes several Scala entry points that can be
        accessed from PySpark once the corresponding JAR is on the classpath.  A
        portable integration is unfortunately tricky without having the library
        available at runtime, so this helper only verifies that the classes are
        reachable and falls back to the exact computation otherwise.  The
        routine is structured in a way that advanced users can monkey patch it to
        call their custom Scala helpers without modifying the rest of the class.
        """

        if self._spark is None:
            self._log(
                "SparkSession is not available; com.nosto.spartann integration "
                "requires a SparkSession for DataFrame interoperability."
            )
            return None

        try:
            # Simply accessing the attribute is enough to raise an AttributeError
            # when the package is not present on the classpath.
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
