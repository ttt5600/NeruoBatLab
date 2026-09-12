"""zfeval — a reusable evaluation suite for zebra finch vocalization models.

Point it at a checkpoint and a dataset registry; get back AUC, accuracy, embedding geometry,
clustering, UMAP, event-level onset/offset scores, baselines, controls, and a comparison against
a previous run.

The design is shaped by bugs that actually happened on this project, so several things that would
normally be optional are enforced:

  * a span that nobody listened to is never scored as negative        (validate.annotated_span)
  * a checkpoint that does not fully load aborts instead of scoring   (features.load_encoder)
  * every metric carries its split and its majority-class rate        (metrics.Score)
  * every comparison carries a bootstrap interval                     (metrics.paired_bootstrap)
  * baselines run every time, so no number is reported alone          (features.mel/energy)
  * every control declares its own null and fails if the null is wrong (controls.*)
"""
__version__ = "0.1.0"
