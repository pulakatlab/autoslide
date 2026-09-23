# Color/Hue Pre-Flight Screening

## Overview

Masson's Trichrome staining hue can drift between scan/staining batches
(reagent lot, scan session, even scanner software or export settings).
This module answers, before running the full pipeline on a new batch:
**does this batch's staining hue look like what the vessel-detection model
was trained on?** It is a pre-flight check, not a correction step -- see
[Color Correction](color-correction.md) for the tool that *transforms*
images toward a reference.

Tracking issue: [#133](https://github.com/pulakatlab/autoslide/issues/133).
Related prior art: `compute_color_divergence`/`compare_histograms`,
implemented in #81/#84 and later deleted during a refactor -- this module
supersedes both with a proper population-level (not pairwise image-vs-image)
statistical treatment.

## How it works

1. **Feature extraction** (`autoslide.src.utils.color_qc.features`): each
   ROI image is reduced to a compact color/hue feature vector on
   tissue-only pixels (LAB mean/std, HSV saturation/value, and a
   Trichrome-specific collagen:muscle log-ratio reusing the same hue band
   `calc_fibrosis.py` already uses to detect fibrosis).
2. **Per-slide aggregation**: features are averaged per *slide*, not per
   ROI image. ROI crops from the same slide are correlated, not
   independent -- fitting a reference distribution directly on hundreds of
   correlated crops understates real between-slide variability and would
   flag ordinary new slides as anomalous.
3. **Reference model** (`autoslide.src.utils.color_qc.reference_model`): a
   robust-center, shrinkage-covariance (`sklearn.covariance.LedoitWolf`)
   Gaussian over per-slide features, scored via Mahalanobis distance. A
   `LedoitWolf` shrinkage estimator was chosen over a raw sample covariance
   or a Gaussian Mixture Model because the training set has only ~10-13
   independent slides -- not enough to identify a multi-component mixture,
   and a raw covariance is poorly conditioned at that sample size.
4. **Screening** (`autoslide.src.utils.color_qc.screen`): groups a new
   batch's images by slide (mirroring the
   `suggested_regions/<SVS_NAME>/images/` convention), scores each slide
   and the batch overall, and reports a 3-tier verdict (`in_bounds`,
   `elevated`, `out_of_bounds`) plus which features drove the score.

## An honest validation caveat

The 13 training slides span (per each slide's Aperio `ImageDescription`
TIFF tag) at least two acquisition sessions on the same scanner/software
(Sep 2021, Mar 2024) and 3 slides with a genuinely different signature
(different Aperio version, MPP, and JPEG-vs-J2K compression). We expected
those 3 slides to be a ready-made positive control for validating this
module's discriminative power. **They are not**: their color/hue feature
values fall in the middle of the other 10 slides' range, not at the edge
-- this dataset simply does not contain a real example of hue drift, and
the old (deleted) pairwise divergence metric shows the identical
non-separation. See
`autoslide_analysis/src/color_qc_validation.py` for the full leave-one-
batch-out analysis and `autoslide_analysis/plots/color_qc/` for the
evidence.

Practical consequence: the shipped default thresholds are calibrated from
the *normal* same-pipeline leave-one-out spread alone (the largest distance
an ordinary held-out slide showed, doubled for the `out_of_bounds`
boundary) -- not from a validated separation against a real drifted batch.
The unit tests confirm the statistical machinery itself correctly
discriminates a large synthetic shift; what is *not* yet proven is
sensitivity to the specific, subtle kind of real-world drift this dataset
happens to contain. Re-run the validation script and update
`default_reference_model.json` once a genuinely different-looking batch is
encountered in real use.

## Usage

```bash
# Screen a batch that mirrors suggested_regions/<SVS_NAME>/images/
# (each subdirectory is treated as one slide)
python -m autoslide.src.utils.color_qc.cli screen \
    --input-dir data/new_batch

# Screen a flat directory of images (treated as a single slide)
python -m autoslide.src.utils.color_qc.cli screen \
    --input-dir data/new_batch/images \
    --json-output report.json

# Screen against a custom (re-fit) reference model
python -m autoslide.src.utils.color_qc.cli screen \
    --input-dir data/new_batch \
    --reference-model path/to/reference_model.json
```

```python
from autoslide.src.utils.color_qc import screen_batch, format_report

report = screen_batch('data/new_batch')
print(format_report(report))
if report['batch_overall']['verdict'] == 'out_of_bounds':
    ...  # flag for manual review before trusting predictions
```

## Pipeline integration

`run_pipeline.py` accepts an optional, **non-fatal** `--check-color-qc`
flag: if set, it screens the suggested-regions directory before running
prediction and logs a warning (not an error -- the pipeline still runs)
if the batch is flagged. Default is off, so existing invocations are
unaffected.

## Why not a Gaussian Mixture Model, MinCovDet, or a raw pixel-histogram?

See `autoslide_analysis` issue #4 for a related (but distinct) proposal to
fit a GMM to whole-tissue pixel color. For *this* module specifically:

- **GMM with >1 component**: unidentifiable at n~10-13 independent slides
  (a 2-component full-covariance GMM in ~8 dimensions needs far more
  parameters than data points).
- **MinCovDet** (automatic outlier trimming): the wrong tool here -- we
  *know* which slides have a different acquisition signature and exclude
  them from the reference by an explicit, auditable choice, rather than
  hoping a trimming algorithm discovers the right subset.
- **Raw per-channel pixel histograms**: far too high-dimensional to fit a
  covariance over at this sample size; the engineered ~8-feature vector is
  a deliberately compact, interpretable summary instead.
