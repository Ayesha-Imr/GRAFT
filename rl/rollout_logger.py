"""graft-monitors: log EVERY training rollout to parquet (seed doc §7.1 fields).

Enabled by GRAFT_ROLLOUT_LOG_DIR (a directory, or "auto" for <output_dir>/rollouts).
One file per rollout: <dir>/step_<global_step>.parquet. Run identity comes from GRAFT_RUN_ID / GRAFT_ARM /
GRAFT_SEED / GRAFT_CONFIG_HASH, set by the launcher. Countdown hack types here are
GRAFT's partition from the reward caches; the seed-doc labeler (with the reuse probe)
is re-run offline on the logged completions.
"""
from __future__ import annotations

import math
import os
import time

import torch

_HACK_COMPONENTS = {
    "countdown_hack_test_overwrite_from_all": "trivial_test",
    "countdown_hack_constraint_relax_from_all": "constraint_removal",
    "countdown_hack_modify_input_from_all": "problem_substitution",
    "countdown_hack_other_from_all": "other",
}


def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def log_rollouts(trainer, output, inputs):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rh_detectors

    t0 = time.time()
    out_dir = os.environ["GRAFT_ROLLOUT_LOG_DIR"]
    if out_dir == "auto":  # sweeps run several trainings in one process: one dir per run
        out_dir = os.path.join(trainer.args.output_dir, "rollouts")
    os.makedirs(out_dir, exist_ok=True)
    n = output["completion_ids"].shape[0]
    step = int(trainer.state.global_step)
    n_coh_tail = trainer._last_coh_n if getattr(trainer, "_coh_env_rows", None) is not None else 0
    n_coh_tail = n_coh_tail or 0
    n_target = n - n_coh_tail

    completions = trainer.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
    comp_lens = output["completion_mask"].sum(dim=1).tolist()
    # Monitor flags for every arm: GRAFT stores them in output, RP/filter only on the trainer.
    is_rh = output.get("is_rh")
    if is_rh is None:
        is_rh = getattr(trainer, "_last_is_rh", None)
    trainer._last_is_rh = None
    is_rh = is_rh.tolist() if is_rh is not None and len(is_rh) == n else [False] * n
    is_coh = output.get("is_coherence")
    is_coh = is_coh.tolist() if is_coh is not None else [False] * n
    adv = output.get("advantages")
    adv = adv.tolist() if adv is not None else [None] * n
    try:
        raw = trainer._reconstruct_raw_rewards().tolist()
    except Exception:
        raw = [None] * n

    comp = {}
    cr = trainer._find_combined_reward()
    if cr is not None:
        for name, fn, _, _ in cr.components:
            sc = fn._last_scores
            if sc is not None and len(sc) in (n, n_target):
                comp[name] = list(sc) + [float("nan")] * (n - len(sc))
    coh_comp = {}
    coh_reward = getattr(trainer, "_coh_reward", None)
    if coh_reward is not None and n_coh_tail:
        for name, fn, _, _ in coh_reward.components:
            sc = getattr(fn, "_last_scores", None)
            if sc is not None and len(sc) == n_coh_tail:
                coh_comp[name] = [float("nan")] * n_target + list(sc)

    try:
        forget_raw, _, _ = trainer._forget_emission_scores(cr)
        forget_raw = forget_raw.tolist() if forget_raw is not None else [None] * n
    except Exception:
        forget_raw = [None] * n
    forget_raw = list(forget_raw) + [None] * (n - len(forget_raw))

    mon = rh_detectors.LAST_MONITOR
    mon_scores = mon["scores"] if mon["scores"] is not None and len(mon["scores"]) == n_target else None
    gr = bool(trainer.gradient_routing_enabled)

    def col(key, i):
        v = inputs[i].get(key) if isinstance(inputs[i], dict) else None
        return v

    rows = {k: [] for k in (
        "run_id", "arm", "seed", "config_hash", "step", "row", "env", "prompt_id", "nums", "target",
        "hackable", "completion", "completion_tokens", "raw_reward", "advantage", "proxy_reward",
        "true_reward", "true_tested", "hack", "hack_type_repo", "forget_reward", "anchor_reward",
        "monitor_score", "monitor_tau", "flagged", "routed", "is_coherence", "sampling_adapters")}
    run_id = os.environ.get("GRAFT_RUN_ID", os.path.basename(trainer.args.output_dir.rstrip("/")))
    arm = os.environ.get("GRAFT_ARM", "")
    seed = int(os.environ.get("GRAFT_SEED", trainer.args.seed))
    chash = os.environ.get("GRAFT_CONFIG_HASH", "")
    for i in range(n):
        # Anchor rows: the cross-env tail (Countdown) or interlaced same-env coherence rows (toy).
        target_row = i < n_target and not bool(is_coh[i])
        hack_type = None
        if target_row:
            for cname, t in _HACK_COMPONENTS.items():
                if cname in comp and (comp[cname][i] or 0) > 0.5:
                    hack_type = t
                    break
        rows["run_id"].append(run_id)
        rows["arm"].append(arm)
        rows["seed"].append(seed)
        rows["config_hash"].append(chash)
        rows["step"].append(step)
        rows["row"].append(i)
        rows["env"].append("target" if target_row else "anchor")
        pid = col("id", i) if target_row else (col("task_id", i) or col("id", i))
        rows["prompt_id"].append(None if pid is None else str(pid))
        nums = col("nums", i) if target_row else None
        rows["nums"].append(None if nums is None else [int(x) for x in nums])
        tgt = col("target", i) if target_row else None
        rows["target"].append(None if tgt is None else int(tgt))
        hk = col("hackable", i) if target_row else None
        rows["hackable"].append(None if hk is None else bool(hk))
        rows["completion"].append(completions[i])
        rows["completion_tokens"].append(int(comp_lens[i]))
        rows["raw_reward"].append(_num(raw[i]))
        rows["advantage"].append(_num(adv[i]))
        rows["proxy_reward"].append(_num(comp.get("countdown_proxy_from_all", [None] * n)[i]))
        rows["true_reward"].append(_num(comp.get("countdown_true_from_all", [None] * n)[i]))
        rows["true_tested"].append(_num(comp.get("countdown_true_tested_from_all", [None] * n)[i]))
        h = comp.get("countdown_hack_from_all")
        rows["hack"].append(None if h is None or not target_row else bool(h[i] > 0.5))
        rows["hack_type_repo"].append(hack_type)
        rows["anchor_reward"].append(_num(coh_comp.get("leetcode_v_correct_from_all", [None] * n)[i]))
        rows["monitor_score"].append(_num(mon_scores[i]) if (mon_scores and i < n_target) else None)
        rows["monitor_tau"].append(mon["tau"] if mon_scores else None)
        rows["forget_reward"].append(_num(forget_raw[i]) if target_row else None)
        rows["flagged"].append(bool(is_rh[i]) and target_row)
        rows["routed"].append(bool(is_rh[i]) and target_row and gr)
        rows["is_coherence"].append(bool(is_coh[i]) or not target_row)
        rows["sampling_adapters"].append(("R+F" if target_row else "R") if gr else "full")
    rh_detectors.LAST_MONITOR["scores"] = None  # consumed; never reused by a later step
    table = pa.table(rows)
    pq.write_table(table, os.path.join(out_dir, f"step_{step:05d}.parquet"), compression="zstd")
    m = trainer._metrics.setdefault("train", {}) if hasattr(trainer, "_metrics") else {}
    if mon_scores is not None:
        m.setdefault("timing/monitor", []).append(float(mon["seconds"]))
        m.setdefault("monitor/score_mean", []).append(float(sum(mon_scores) / len(mon_scores)))
    m.setdefault("timing/rollout_log", []).append(time.time() - t0)
