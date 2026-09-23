"""
Population-level reference model for the color_qc pre-flight screening
module: fits a robust multivariate reference distribution over per-slide
color/hue features (see ``features.py``) from known-good training slides,
and scores new slides/batches against it via Mahalanobis distance.

Deliberately NOT a Gaussian Mixture Model or a MinCovDet-based automatic
outlier-trimming estimator -- see docs/advanced/color-qc-screening.md for
the statistical rationale (in short: n~10-13 independent slides is too few
to identify a multi-component mixture, and any known-different slides
should be *excluded* from the reference by an explicit, auditable choice
rather than left in for a trimming algorithm to discover).
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

from autoslide.src.utils.color_qc.features import FEATURE_NAMES


class ReferenceColorModel:
    """Robust-center + shrinkage-covariance Gaussian reference model.

    Fit once on a set of per-slide feature dicts (see
    ``features.aggregate_slide_features``) drawn from known-good
    acquisition sessions, then used to score new slides via Mahalanobis
    distance.
    """

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES):
        self.feature_names = list(feature_names)
        self.center_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.n_slides_fit_: int = 0
        self.thresholds_: Dict[str, float] = {}

    def _to_matrix(self, slide_features: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.array(
            [[sf[name] for name in self.feature_names]
                for sf in slide_features],
            dtype=np.float64,
        )

    def fit(self, slide_features: Sequence[Dict[str, float]]) -> 'ReferenceColorModel':
        """Fit the reference model on a list of per-slide feature dicts.

        Center is the per-feature median (robust to a single unusual
        slide); covariance is Ledoit-Wolf shrinkage (well-conditioned even
        when n is close to, or smaller than, the number of features).
        """
        X = self._to_matrix(slide_features)
        if X.shape[0] < 3:
            raise ValueError(
                f"Need at least 3 slides to fit a reference model, got {X.shape[0]}")

        self.center_ = np.median(X, axis=0)
        lw = LedoitWolf().fit(X - self.center_)
        self.covariance_ = lw.covariance_
        self.precision_ = np.linalg.pinv(self.covariance_)
        self.n_slides_fit_ = X.shape[0]
        return self

    def mahalanobis(self, slide_feature: Dict[str, float]) -> float:
        """Squared Mahalanobis distance of one slide's feature vector from
        the reference center."""
        if self.center_ is None:
            raise RuntimeError("ReferenceColorModel is not fit yet")
        x = np.array([slide_feature[name]
                      for name in self.feature_names], dtype=np.float64)
        delta = x - self.center_
        return float(delta @ self.precision_ @ delta)

    def feature_contributions(self, slide_feature: Dict[str, float]) -> Dict[str, float]:
        """Per-feature robust z-score-like contribution, for the diagnostic
        report. Uses the marginal (diagonal) standard deviation from the
        fitted covariance -- a simplification of the full Mahalanobis
        decomposition, chosen because it is directly interpretable
        ("feature X is N sigma from the reference") rather than a coupled
        quadratic-form term.
        """
        if self.center_ is None:
            raise RuntimeError("ReferenceColorModel is not fit yet")
        x = np.array([slide_feature[name]
                      for name in self.feature_names], dtype=np.float64)
        delta = x - self.center_
        marginal_std = np.sqrt(np.clip(np.diag(self.covariance_), 1e-12, None))
        z = delta / marginal_std
        return dict(zip(self.feature_names, z.tolist()))

    def approx_chi2_tail(self, d2: float) -> float:
        """Approximate chi-square(p) upper-tail probability for a squared
        Mahalanobis distance. This is an asymptotic, Gaussian-assumption
        guide -- NOT a calibrated p-value at n~10-13 slides. Use
        ``score()``'s empirically-calibrated verdict tiers for decisions.
        """
        return float(stats.chi2.sf(d2, df=len(self.feature_names)))

    def set_thresholds(self, in_bounds_max: float, elevated_max: float) -> None:
        """Set the empirically-calibrated verdict-tier boundaries (see the
        leave-one-batch-out validation in
        ``autoslide_analysis/src/color_qc_validation.py`` for how these
        should be derived from real data)."""
        if not in_bounds_max < elevated_max:
            raise ValueError("in_bounds_max must be < elevated_max")
        self.thresholds_ = {
            'in_bounds_max': float(in_bounds_max),
            'elevated_max': float(elevated_max),
        }

    def score(self, slide_feature: Dict[str, float]) -> Dict[str, object]:
        """Score one slide's feature vector against the fitted reference.

        Returns a dict with the raw distance, an approximate chi-square
        annotation, a per-feature breakdown, and (if thresholds have been
        set via ``set_thresholds``) a categorical verdict.
        """
        d2 = self.mahalanobis(slide_feature)
        contributions = self.feature_contributions(slide_feature)
        top_features = dict(
            sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:3])

        result: Dict[str, object] = {
            'mahalanobis_sq': d2,
            'approx_chi2_tail_probability': self.approx_chi2_tail(d2),
            'top_feature_contributions': top_features,
            'n_slides_in_reference': self.n_slides_fit_,
        }

        if self.thresholds_:
            if d2 <= self.thresholds_['in_bounds_max']:
                verdict = 'in_bounds'
            elif d2 <= self.thresholds_['elevated_max']:
                verdict = 'elevated'
            else:
                verdict = 'out_of_bounds'
            result['verdict'] = verdict
            result['thresholds'] = dict(self.thresholds_)
        else:
            result['verdict'] = 'unknown (no thresholds calibrated)'

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model as plain JSON (no pickle, to avoid a
        numpy/pickle-version-compat footgun for an artifact that may be
        regenerated on one machine and read on another)."""
        if self.center_ is None:
            raise RuntimeError("ReferenceColorModel is not fit yet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'feature_names': self.feature_names,
            'center': self.center_.tolist(),
            'covariance': self.covariance_.tolist(),
            'n_slides_fit': self.n_slides_fit_,
            'thresholds': self.thresholds_,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ReferenceColorModel':
        with open(path, 'r') as f:
            payload = json.load(f)
        model = cls(feature_names=payload['feature_names'])
        model.center_ = np.array(payload['center'], dtype=np.float64)
        model.covariance_ = np.array(payload['covariance'], dtype=np.float64)
        model.precision_ = np.linalg.pinv(model.covariance_)
        model.n_slides_fit_ = payload['n_slides_fit']
        model.thresholds_ = payload.get('thresholds', {})
        return model
