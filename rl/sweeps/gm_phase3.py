"""graft-monitors pilot Phase 3 (seed doc §7.6): P1-P4, 1 seed each, paper hyperparameters.

  P1 GRAFT + keyword   = sweeps/countdown_hf100_gr_lccoh64_lr3.py        (paper GRAFT arm)
  P2 RP + keyword      = sweeps/countdown_hf100_rp_lccoh64.py rp2_lc64_lr1 (paper BEST_RP)
  P3 RP + decision     = P2 with configs/gm_countdown_decision.yaml
  P4 GRAFT + decision  = P1 with configs/gm_countdown_decision.yaml
Only change from the paper configs: save_steps 50 (checkpoints for post-hoc evals at
0/50/100/150/200). Select one arm with GM_ARM=P1..P4 (one run per H200 Pod).
"""
import os

from sweeps.countdown_hf100_gr_lccoh64_lr3 import runs as _gr
from sweeps.countdown_hf100_rp_lccoh64 import runs as _rp

_SEED = 9
_P1 = next(r for r in _gr if r["seed"] == _SEED)
_P2 = next(r for r in _rp if r["run_name"] == f"cdhf100_rp2_lc64_lr1_s{_SEED}")
_DEC = "configs/gm_countdown_decision.yaml"
_ARMS = {
    "P1": {**_P1, "run_name": f"gm_P1_graft_keyword_s{_SEED}"},
    "P2": {**_P2, "run_name": f"gm_P2_rp_keyword_s{_SEED}"},
    "P3": {**_P2, "config": _DEC, "run_name": f"gm_P3_rp_decision_s{_SEED}"},
    "P4": {**_P1, "config": _DEC, "run_name": f"gm_P4_graft_decision_s{_SEED}"},
}
for _r in _ARMS.values():
    _r["save_steps"] = 50

_sel = os.environ.get("GM_ARM")
runs = [_ARMS[_sel]] if _sel else list(_ARMS.values())
per_gpu = 1
no_baseline = True
