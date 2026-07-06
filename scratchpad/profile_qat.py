"""Profile ONE q56s-style QAT training step to find where the ~1.07 s/step (KD) goes.
GPU util during training oscillates 18-91% (mean ~50%) at 64 W -> roughly half the step is
stall. Suspects: (a) the inherently-sequential WKV kernels (B*H*K lanes, tiny model) that
cannot fill the 4070, (b) per-step Python/launch overhead (KD teacher, EMA loop, optimizer),
(c) the anneal soft-phase softmax path. This script replicates the EXACT q56s lever stack and
measures: phase-level wall split (explicit cuda.synchronize around segments, unprofiled steps)
+ torch.profiler kernel tables for the anneal SOFT phase and the HARD phase separately.
Run from a wrapper cmd AFTER the GPU frees (waits are in the cmd, not here).
Output: scratchpad/profile_qat_out.txt (+ chrome traces prof_qat_{soft,hard}.json.gz).
"""
import os
import sys
import time

ROOT = r"C:\Users\Andrew\rwkv-state-quant"
GT = os.path.join(ROOT, "gpu_train")
SCRATCH = os.path.join(ROOT, "scratchpad")
# RWKV_PROFILE_TAG: suffix for output files (e.g. "_fast" for the speed-flag A/B rerun) so runs
# don't clobber each other. The speed flags themselves (RWKV_QAT_ROT_CACHE / RWKV_QAT_FAST_EMB /
# RWKV_QAT_EMA_FOREACH) are inherited from the caller's environment — NOT forced here.
TAG = os.environ.get("RWKV_PROFILE_TAG", "")

# --- env: EXACT q56s lever stack. MUST precede any rwkv import (module-level env reads). ---
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.update({
    "RWKV_QAT_PQ": os.path.join(SCRATCH, "pq_cb_m1b5.txt"),
    "RWKV_QAT_PQ_LEARN": "1",
    "RWKV_QAT_SHIFT_PQ": os.path.join(SCRATCH, "pq_cb_shift_m4b4.txt"),
    "RWKV_QAT_SHIFT_PQ_LEARN": "1",
    "RWKV_QAT_SHIFT_ROT": "1",
    "RWKV_QAT_NORM_BITS": "1",
    "RWKV_QAT_SHIFT_ANNEAL": "0.05",
    "RWKV_QAT_KD": "0.2",
    "RWKV_QAT_CB_RESURRECT": "1",
    "RWKV_WEIGHT_DECAY": "0.01",
    "RWKV_CLIP": "0.25",
    "RWKV_EMA_DECAY": "0.99",
    "RWKV_N_HEADS": "2",
    "RWKV_HEAD_DIM": "16",
    "RWKV_NO_JIT": "1",
    "RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",
    "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3",
    "RWKV_QAT_FUSED": "1",
    "RWKV_EMPTY_CACHE_EVERY": "0",
    "RWKV_DETERMINISTIC": "1",
    "RWKV_AUGMENT_SEED": "1234",
    "OMP_NUM_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
})
os.chdir(GT)
sys.path.insert(0, GT)
sys.argv = [sys.argv[0], "--config", "configs/qat_pq_q56s.toml"]

import multiprocessing

N_FETCH = 4
WARM = 6          # eager warmup steps (first-launch inits, allocator)
SEG = 5           # unprofiled steps with explicit per-segment sync timing
PROF = 8          # profiled steps per phase


def make_step(config, master_model, model, teacher, optimizer, scheduler,
              shift_cb_param, shift_rot_param, wkv_cb_param, ema_state, res_ema, rwkv_ops_mod):
    """One production-faithful training step over `batch`. Returns dict of segment seconds
    (segments only synced when `timed`)."""
    import torch
    from rwkv.train_rwkv import transfer_child_grad_to_master
    kd_lam = 0.2
    ema_decay = 0.99
    RES_DECAY = 0.99

    from rwkv.model import rwkv_model as _rm_rc
    ema_foreach = os.environ.get("RWKV_QAT_EMA_FOREACH", "") == "1"
    ema_lists = [None]

    def step_fn(batch, timed=False):
        segs = {}

        def mark(name, t0):
            if timed:
                torch.cuda.synchronize()
                segs[name] = time.perf_counter() - t0
            return time.perf_counter()

        _rm_rc.shift_rot_cache_clear()  # per-step hygiene, mirrors train_rwkv (no-op when cache off)
        t0 = time.perf_counter()
        model.copy_downcast_(master_model, dtype=config.DTYPE)
        model.train()
        t0 = mark("copy_downcast", t0)

        with torch.no_grad():
            t_ahead, t_w, _t_wlp, t_p = teacher.forward_batch(
                batch.start, batch.sub_gather, batch.sub_gather_lens,
                batch.time_shift_selects, batch.skips, batch.num_data,
            )
            _les = batch.labels.float()[..., 0].unsqueeze(-1)
            _tcr = teacher.forgetting_curve(t_w, _les).clamp(1e-5, 1 - 1e-5)
            _tcl = torch.log(_tcr / (1 - _tcr)) + teacher.interp(t_ahead, _les)
            kd_args = (t_p.float(), torch.sigmoid(_tcl).clamp(1e-5, 1 - 1e-5).float(), kd_lam)
        t0 = mark("teacher_fwd", t0)

        stats = model.get_loss(batch, kd=kd_args)
        assert stats is not None and torch.isfinite(stats.average_loss)
        t0 = mark("student_fwd", t0)

        rwkv_ops_mod.wkv_pq_grad_zero()
        stats.average_loss.backward()
        t0 = mark("backward", t0)

        transfer_child_grad_to_master(master=master_model, child=model)
        rwkv_ops_mod.wkv_pq_grad_fetch()
        _clip_params = list(master_model.parameters()) + [shift_cb_param, shift_rot_param, wkv_cb_param]
        total_norm = torch.nn.utils.clip_grad_norm_(_clip_params, 0.25)
        assert torch.isfinite(total_norm)
        optimizer.step()
        rwkv_ops_mod.wkv_pq_reupload()
        t0 = mark("xfer_clip_optstep", t0)

        # resurrection per-step cost = the grad-norm EMA update (re-seed itself is every 250 steps)
        for name, p, shape in (("shiftcb", shift_cb_param, None), ("wkvcb", wkv_cb_param, None)):
            if p.grad is None:
                continue
            if name == "shiftcb":
                _r2, _m, _nc, _sd = p.shape
                gn = p.grad.detach().float().view(_r2 * _m, _nc, _sd).norm(dim=-1)
            else:
                _m, _sd, _nc = rwkv_ops_mod._PQ_META[0], rwkv_ops_mod._PQ_META[1], rwkv_ops_mod._PQ_META[2]
                gn = p.grad.detach().float().view(2 * _m, _nc, _sd).norm(dim=-1)
            ema = res_ema.get(name)
            res_ema[name] = gn.clone() if ema is None else RES_DECAY * ema + (1 - RES_DECAY) * gn
        optimizer.zero_grad()
        t0 = mark("resurrect_zero", t0)

        msd = master_model.state_dict()
        if not ema_state:
            for k, v in msd.items():
                if v.is_floating_point():
                    ema_state[k] = v.detach().float().clone()
        elif ema_foreach:
            if ema_lists[0] is None:
                ema_lists[0] = ([ema_state[k] for k in msd if k in ema_state],
                                [msd[k].detach() for k in msd if k in ema_state])
            torch._foreach_mul_(ema_lists[0][0], ema_decay)
            torch._foreach_add_(ema_lists[0][0], ema_lists[0][1], alpha=1 - ema_decay)
        else:
            for k, v in msd.items():
                if k in ema_state:
                    ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1 - ema_decay)
        scheduler.step()
        t0 = mark("ema_sched", t0)
        return segs

    return step_fn


def run(task_queue, batch_queue):
    import torch
    import numpy as np
    from rwkv.parse_toml import parse_toml
    from rwkv.train_rwkv import get_optimizer, get_groups, _maybe_enable_determinism, maybe_compile_mixers
    from rwkv.data_fetcher import DataFetcher
    from rwkv.model.srs_model import SrsRWKV
    from rwkv.model import rwkv_model as rwkv_model_mod
    from rwkv.model import rwkv_ops as rwkv_ops_mod
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

    config = parse_toml()
    _maybe_enable_determinism()
    out_path = os.path.join(SCRATCH, f"profile_qat_out{TAG}.txt")
    emit_flags = {k: os.environ.get(k, "") for k in
                  ("RWKV_QAT_ROT_CACHE", "RWKV_QAT_FAST_EMB", "RWKV_QAT_EMA_FOREACH")}
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(str(s))

    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).selective_cast(config.DTYPE).to(config.DEVICE)
    optimizer = get_optimizer(config, master_model)
    model_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}.pth"
    optim_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}_optim.pth"
    master_model.load_state_dict(torch.load(model_path, weights_only=True))
    _intended_wd = [g["weight_decay"] for g in optimizer.param_groups]
    optimizer.load_state_dict(torch.load(optim_path, weights_only=False))
    for _g, _wd in zip(optimizer.param_groups, _intended_wd):
        _g["weight_decay"] = _wd
    for _g in optimizer.param_groups:
        _g["lr"] = config.PEAK_LR
        _g["initial_lr"] = config.PEAK_LR
    model.copy_downcast_(master_model, dtype=config.DTYPE)

    teacher = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    teacher.load_state_dict(torch.load(model_path, weights_only=True))
    teacher = teacher.selective_cast(config.DTYPE).to(config.DEVICE)
    for _m in teacher.modules():
        if getattr(_m, "state_shift_qmax", float("inf")) != float("inf"):
            _m.state_shift_qmax = float("inf")
        if getattr(_m, "state_qmax", float("inf")) != float("inf"):
            _m.state_qmax = float("inf")
        if getattr(_m, "state_lowrank_rank", 0) > 0:
            _m.state_lowrank_rank = 0
    teacher.eval()
    for _p in teacher.parameters():
        _p.requires_grad_(False)
    maybe_compile_mixers(model, "(student)")
    maybe_compile_mixers(teacher, "(teacher)")

    shift_cb_param = rwkv_model_mod.shift_pq_init(config.DEVICE)
    optimizer.add_param_group({"params": [shift_cb_param], "lr": config.PEAK_LR, "weight_decay": 0.0})
    shift_rot_param = rwkv_model_mod.shift_rot_init(config.DEVICE, 32)
    optimizer.add_param_group({"params": [shift_rot_param], "lr": config.PEAK_LR, "weight_decay": 0.0})
    wkv_cb_param = rwkv_ops_mod.wkv_pq_cb_param()
    optimizer.add_param_group({"params": [wkv_cb_param], "lr": config.PEAK_LR, "weight_decay": 0.0})

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: 1 + np.cos(0.5 * np.pi * (1 + t / 1000)))

    groups = get_groups(config.TRAIN_DATASET_LMDB_PATH, config.TRAIN_DATASET_LMDB_SIZE,
                        config.MAX_TRAIN_GLOBAL_LEN, users=list(range(1000, 1300)))
    n_need = WARM + SEG + PROF + 2 + PROF + 2
    for i in range(n_need):
        data_fetcher.enqueue((f"train-{i}", groups[i % len(groups)]))
    emit(f"[profile] {n_need} batches enqueued ({N_FETCH} fetch procs), device={config.DEVICE}")
    emit(f"[profile] speed flags: {emit_flags}")

    ema_state, res_ema = {}, {}
    step_fn = make_step(config, master_model, model, teacher, optimizer, scheduler,
                        shift_cb_param, shift_rot_param, wkv_cb_param, ema_state, res_ema, rwkv_ops_mod)

    def get_batch(i):
        b = data_fetcher.get(f"train-{i}")
        return b.to(config.DEVICE)

    bi = 0
    # ---------- warmup (SOFT phase: progress 0.25 -> tau > 0) ----------
    rwkv_model_mod.set_shift_anneal_progress(0.25)
    for _ in range(WARM):
        step_fn(get_batch(bi)); bi += 1
    torch.cuda.synchronize()

    # ---------- segment timing, unprofiled (soft phase) ----------
    seg_tot = {}
    wall = []
    for _ in range(SEG):
        t0 = time.perf_counter()
        segs = step_fn(get_batch(bi), timed=True); bi += 1
        wall.append(time.perf_counter() - t0)
        for k, v in segs.items():
            seg_tot[k] = seg_tot.get(k, 0.0) + v
    emit("\n===== SEGMENT SPLIT (soft anneal phase, mean over %d steps, synced) =====" % SEG)
    tot = sum(seg_tot.values())
    for k, v in sorted(seg_tot.items(), key=lambda kv: -kv[1]):
        emit(f"  {k:20s} {v/SEG*1000:8.1f} ms  ({v/tot*100:5.1f}%)")
    emit(f"  {'TOTAL':20s} {tot/SEG*1000:8.1f} ms   (wall mean {sum(wall)/len(wall)*1000:.1f} ms)")

    # ---------- profiler, SOFT phase ----------
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_soft:
        for _ in range(PROF):
            step_fn(get_batch(bi)); bi += 1
        torch.cuda.synchronize()
    emit("\n===== PROFILER soft phase: top 30 by CUDA total =====")
    emit(prof_soft.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    emit("\n===== PROFILER soft phase: top 15 by CPU total =====")
    emit(prof_soft.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    try:
        prof_soft.export_chrome_trace(os.path.join(SCRATCH, f"prof_qat_soft{TAG}.json.gz"))
    except Exception as e:
        emit(f"[trace export soft failed: {e}]")
    del prof_soft

    # ---------- profiler, HARD phase (progress 0.75 -> tau == 0, argmin path) ----------
    rwkv_model_mod.set_shift_anneal_progress(0.75)
    for _ in range(2):
        step_fn(get_batch(bi)); bi += 1  # re-warm the hard path
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_hard:
        for _ in range(PROF):
            step_fn(get_batch(bi)); bi += 1
        torch.cuda.synchronize()
    emit("\n===== PROFILER hard phase: top 30 by CUDA total =====")
    emit(prof_hard.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    emit("\n===== PROFILER hard phase: top 15 by CPU total =====")
    emit(prof_hard.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    try:
        prof_hard.export_chrome_trace(os.path.join(SCRATCH, f"prof_qat_hard{TAG}.json.gz"))
    except Exception as e:
        emit(f"[trace export hard failed: {e}]")

    with open(out_path, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    emit(f"\n[profile] written {out_path}")


def main():
    from rwkv.parse_toml import parse_toml
    from rwkv.prepare_batch import prepare_data_train_test
    config = parse_toml()
    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()
        procs = []
        for _ in range(N_FETCH):
            p = multiprocessing.Process(
                target=prepare_data_train_test,
                args=(config.TRAIN_DATASET_LMDB_PATH, config.TRAIN_DATASET_LMDB_SIZE,
                      config.VALIDATE_DATASET_LMDB_PATH, config.VALIDATE_DATASET_LMDB_SIZE,
                      task_queue, batch_queue, config.MAX_TRAIN_GLOBAL_LEN, 1234),
            )
            p.start()
            procs.append(p)
        try:
            run(task_queue, batch_queue)
        finally:
            for p in procs:
                p.terminate()
            print("Killed processes.")


if __name__ == "__main__":
    main()
