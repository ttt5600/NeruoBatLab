#!/usr/bin/env python
"""Record run15's realised compute next to compute_budget.json's run11 and AVES rows.

Every input is a MEASURED value with its source named; the derived fields follow from them.
run16 is included as a PROJECTION and labelled as one -- it has not run.
"""
import json
from pathlib import Path

A = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
updates = 93750                 # checkpoint epoch=9-step=93750.ckpt, global_step 93750 at export
steps_per_epoch = 10032         # Lightning progress bar, job 39132082: "Epoch 0: 0/10032"
corpus_hours = 224.4209         # sum of TSV frame counts / 16000 / 3600, verified on all 763 rows
spu = corpus_hours * 3600 / steps_per_epoch

out = {
    "run15": dict(name="run15_combined", updates=updates, steps_per_epoch=steps_per_epoch,
                  corpus_hours=corpus_hours, audio_seconds_per_update=spu,
                  total_audio_hours=spu * updates / 3600, epochs=updates / steps_per_epoch,
                  world_size_realised=1, num_clusters=200, status="measured",
                  sources=dict(updates="checkpoint global_step",
                               steps_per_epoch="progress bar, job 39132082",
                               corpus_hours="TSV frame-count sum, all 763 rows resolved")),
    "run16_projected": dict(name="run16_compute4x", updates=updates, world_size_planned=4,
                            corpus_hours=corpus_hours, audio_seconds_per_update=4 * spu,
                            total_audio_hours=4 * spu * updates / 3600,
                            status="PROJECTED -- assumes smoke_ddp4 confirms 4-way sharding; "
                                   "has not run"),
}
(A / "run15_compute.json").write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(f"{k:<17} {v['audio_seconds_per_update']:7.1f} s/update  "
          f"{v['total_audio_hours']:8.0f} h total  [{v['status'][:9]}]")
print(f"wrote {A/'run15_compute.json'}")
