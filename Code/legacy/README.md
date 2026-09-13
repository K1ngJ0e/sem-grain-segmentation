# Original source archive

`original_pipeline.py` preserves the author-supplied script, apart from line
endings. It is kept for provenance and comparison only. It has no augmentation,
uses batch-mean validation aggregation, enables manual correction by default,
and does not perform a separate final held-out test in its default workflow.

Use `../run.py` and `../configs/manuscript.json` for the revised implementation.
Results from the revised workflow require a new training/evaluation run and must
not be presented as evidence of the historical settings of this original script.
