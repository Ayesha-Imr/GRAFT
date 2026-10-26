"""graft-monitors post-pilot check 2: reward penalty against a decision-model monitor on object_qa.

The paper's RP(2.0) arm at seed 1 (sweeps/respmon_ivainv_dn_rp_3seed.py), 1000 steps, with the
keyword monitor (blind to "indeed") replaced by AnyJev Tacit-1.7B (configs/gm_object_qa_decision.yaml).
Pilot Phase 1 ran the keyword version: undetected hacking rose after about step 650 (0.09 at 990).
The read is whether undetected hacking also emerges when the monitor is an LLM with partial recall.
"""
from sweeps.respmon_ivainv_dn_rp_3seed import runs as _dnrp

runs = [{**r, "config": "configs/gm_object_qa_decision.yaml",
         "run_name": "object_qa_gm_decision_rp2_hf050_st1000_s1"}
        for r in _dnrp if r["config"] == "configs/object_qa_respmon.yaml" and r["seed"] == 1
        and "_rp2_" in r["run_name"]]
assert len(runs) == 1 and runs[0]["max_steps"] == 1000, runs
per_gpu = 1
no_baseline = True
