1 - Extract regions from the image and ask user to label them
2 - Extract sections from a given region with specified properties:
	- Distance from edge
	- No blood vessels
	- No excessive fibrosis

PIPELINE:
1 - Initial Annotation : Extract regions from the image and ask user to label them
2 - Final Annotation : Save metadata for labelled regions
3 - suggest_regions : Suggest regions for annotation
