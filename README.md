Intially perform a thresholding to extract samples.
Step through extracted samples with given step and window sizes.
Discard frames with:
    1. Too much background
    2. Blood vessels
    3. Bulk (non-interstitial fibrosis)

For surviving samples, perform fibrosis calculation

1 - Extract regions from the image and ask user to label them
2 - Extract sections from a given region with specified properties:
	- Distance from edge
	- No blood vessels
	- No excessive fibrosis

PIPELINE:
1 - Initial Annotation : Extract regions from the image and ask user to label them
2 - Final Annotation : Save metadata for labelled regions
3 - suggest_regions : Suggest regions for annotation
