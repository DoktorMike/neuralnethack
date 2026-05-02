# TODO

## Statistics
- Calculate a P-value for the ROC curve. Bootstrap CI on AUC is the
  cheap version (resample dataset → recompute AUC many times → percentile
  / one-sided p-value). DeLong's test is the principled binary
  comparison; only worth it if a concrete user asks.
