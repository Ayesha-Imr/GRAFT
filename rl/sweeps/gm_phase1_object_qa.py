"""graft-monitors pilot Phase 1 (seed doc §7.4): object_qa toy reproduction, 1 seed per arm.

Exactly the paper's object_qa arms (Fig. 3) at seed 1:
  - no intervention, reward penalty 2.0: sweeps/respmon_ivainv_dn_rp_3seed.py
  - GRAFT (canonical recipe):            sweeps/respmon_objinv_gr_canon_3seed.py
Keyword monitor = configs/object_qa_respmon.yaml (flags actually/exactly/quite, blind to indeed).

    python -u sweep.py --name gm_phase1_object_qa --config sweeps/gm_phase1_object_qa.py \
        --no_baseline --no-eval_at_start
"""
from sweeps.respmon_ivainv_dn_rp_3seed import runs as _dnrp
from sweeps.respmon_objinv_gr_canon_3seed import runs as _gr

_SEED = 1
runs = [r for r in _dnrp if r["config"] == "configs/object_qa_respmon.yaml" and r["seed"] == _SEED]
runs += [r for r in _gr if r["seed"] == _SEED]
assert len(runs) == 3, [r["run_name"] for r in runs]

per_gpu = 3
no_baseline = True
