"""graft-monitors pilot Phase 1 step 3: GRAFT on object_qa with a decision-model monitor.

Pipeline smoke test of the monitor interface inside training (not a science result):
the canonical GRAFT recipe with rh_detector = decision_monitor (AnyJev served on the
same Pod, question "does the answer use an intensifier for emphasis?", tau 0.5 untuned).
200 steps: enough to exercise routing, logging and latency at scale.
"""
from sweeps.respmon_objinv_gr_canon_3seed import runs as _gr

runs = [{**r, "config": "configs/gm_object_qa_decision.yaml", "max_steps": 200,
         "run_name": "object_qa_gm_decision_gr_cls_canon_hf050_st200_s1"}
        for r in _gr if r["seed"] == 1]
assert len(runs) == 1
per_gpu = 1
no_baseline = True
