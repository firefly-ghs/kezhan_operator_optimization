"""任务四-B：双 GPU 拆分投影与通信 benchmark。

把一个 down 投影 y = W x（W [H, I]、x [T, I]）按 K 维切给 2 张卡：
每个 rank 只算 x[:, 半] @ W[:, 半]ᵀ 的 partial，再用 all-reduce / P2P 求和；
另提供按输出列切分 + all-gather 的对照模式。

用法（GPU 机器，双进程 NCCL）：
    torchrun --nproc_per_node=2 run_two_gpu.py --backend nccl --tokens 64 --repeat 3
    python run_two_gpu.py --nproc 2 --backend nccl --tokens 8
CPU 逻辑校验（Windows 的 gloo 不可用，只验证数值与切分逻辑）：
    python run_two_gpu.py --simulate --tokens 8 --hidden 512 --intermediate 1024

注意：`p2p_reduce` 仅支持 world_size=2。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common.utils import (
    TRITON_AVAILABLE,
    bench_callable,
    dtype_from_name,
    dtype_name,
    error_stats,
    get_device,
    print_table,
    save_result,
)

import kernels_tp2


# --------------------------------------------------------------------------- #
# 数据与参考
# --------------------------------------------------------------------------- #
def _make_data(t: int, h: int, i: int, dtype: torch.dtype, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(t, i, generator=gen) * 0.5).to(dtype).to(device)
    w = (torch.randn(h, i, generator=gen) * 0.02).to(dtype).to(device)
    return x, w


def _local(x, w, out, use_triton: bool) -> None:
    if use_triton:
        kernels_tp2.local_projection(x, w, out)
    else:
        torch.mm(x, w.t(), out=out)


def _p2p_sum(t: torch.Tensor) -> None:
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError("p2p_reduce 仅支持 world_size=2")
    peer = 1 - dist.get_rank()
    recv = torch.empty_like(t)
    ops = [
        dist.P2POp(dist.isend, t, peer),
        dist.P2POp(dist.irecv, recv, peer),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    t.add_(recv)


def _local_chunked(x, w, out, use_triton: bool, chunks: int) -> None:
    t = out.shape[0]
    step = max(1, math.ceil(t / chunks))
    handles = []
    for r0 in range(0, t, step):
        r1 = min(r0 + step, t)
        _local(x[r0:r1], w, out[r0:r1], use_triton)
        handles.append(dist.all_reduce(out[r0:r1], op=dist.ReduceOp.SUM, async_op=True))
    for handle in handles:
        handle.wait()


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


# --------------------------------------------------------------------------- #
# 单个 rank 的完整流程
# --------------------------------------------------------------------------- #
def _worker(rank: int, world: int, cfg: dict, init_uri: str) -> None:
    backend = cfg["backend"]
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=backend, init_method=init_uri, rank=rank, world_size=world
    )

    try:
        device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
        dtype = dtype_from_name(cfg["dtype"])
        torch.manual_seed(0)

        t, h, i = cfg["tokens"], cfg["hidden"], cfg["intermediate"]
        x, w = _make_data(t, h, i, dtype, device)

        # 静态显存分解（rank0 汇总打印）
        if rank == 0:

            def _mb(tensor):
                return tensor.numel() * tensor.element_size() / 1e6

            print("静态显存分解(MB/卡): " + "  ".join(
                f"{k}={v:.2f}" for k, v in {
                    "x(T×I)": _mb(x), "w(H×I)": _mb(w),
                    "k_shard": _mb(x) / 2, "w_k_shard": _mb(w) / 2,
                    "w_col_shard": _mb(w) / 2,
                    "out(T×H)": t * h * 4 / 1e6,  # Triton 路径输出为 fp32
                }.items()
            ))

        # K 维切分：每个 rank 只持有自己的半份
        k_half = i // 2
        x_shard = x[:, rank * k_half : (rank + 1) * k_half].contiguous()
        w_shard = w[:, rank * k_half : (rank + 1) * k_half].contiguous()

        # 列切分对照：每个 rank 持有 H/2 行权重，K 维完整
        h_half = h // world
        w_rows = w[rank * h_half : (rank + 1) * h_half].contiguous()

        # float64 参考：K 维 partial 求和 / 全量 matmul
        ref_k = x_shard.double() @ w_shard.double().t()
        dist.all_reduce(ref_k, op=dist.ReduceOp.SUM)
        ref_full = x.double() @ w.double().t()

        atol = cfg["atol"]
        rtol = cfg["rtol"]
        use_triton = backend == "nccl" and TRITON_AVAILABLE
        if backend == "nccl" and not TRITON_AVAILABLE:
            print(f"[rank {rank}] [skip] 未安装 triton，本地 kernel 回退到 torch.mm")

        modes = cfg["modes"]
        single_out = torch.empty(t, h, dtype=dtype, device=device)
        single_fn = lambda: torch.mm(x, w.t(), out=single_out)

        all_metrics = []
        for mode in modes:
            if mode == "allreduce_triton" and not use_triton:
                if rank == 0:
                    print(f"[skip] {mode} 需要 GPU + Triton")
                all_metrics.append({"mode": mode, "skipped": True})
                continue
            if mode == "p2p_reduce" and world != 2:
                if rank == 0:
                    print(f"[skip] {mode} 仅支持 world_size=2")
                all_metrics.append({"mode": mode, "skipped": True})
                continue

            dist.barrier()
            use_tri = mode in ("allreduce_triton", "chunked_overlap", "p2p_reduce") and use_triton
            if rank == 0 and mode in ("chunked_overlap", "p2p_reduce") and not use_tri:
                print(f"[info] {mode} 在当前环境回退为 torch.mm 本地计算")
            out_dtype = torch.float32 if use_tri else dtype
            out = torch.empty(t, h, dtype=out_dtype, device=device)

            if mode == "allreduce_torch":
                local = lambda: _local(x_shard, w_shard, out, False)
                comm = lambda: dist.all_reduce(out, op=dist.ReduceOp.SUM)

                def total():
                    _local(x_shard, w_shard, out, False)
                    dist.all_reduce(out, op=dist.ReduceOp.SUM)

                check_out, check_ref = out, ref_k
            elif mode == "allreduce_triton":
                local = lambda: _local(x_shard, w_shard, out, True)
                comm = lambda: dist.all_reduce(out, op=dist.ReduceOp.SUM)

                def total():
                    _local(x_shard, w_shard, out, True)
                    dist.all_reduce(out, op=dist.ReduceOp.SUM)

                check_out, check_ref = out, ref_k
            elif mode == "chunked_overlap":
                local = lambda: _local(x_shard, w_shard, out, use_tri)
                scratch = torch.empty_like(out)
                comm = lambda: dist.all_reduce(scratch, op=dist.ReduceOp.SUM)
                total = lambda: _local_chunked(x_shard, w_shard, out, use_tri, cfg["chunks"])
                check_out, check_ref = out, ref_k
            elif mode == "p2p_reduce":
                local = lambda: _local(x_shard, w_shard, out, use_tri)
                scratch = torch.empty_like(out)
                comm = lambda: _p2p_sum(scratch)

                def total():
                    _local(x_shard, w_shard, out, use_tri)
                    _p2p_sum(out)

                check_out, check_ref = out, ref_k
            elif mode == "column_parallel":
                y_shard = torch.empty(t, h_half, dtype=dtype, device=device)
                gathered = torch.empty(world, t, h_half, dtype=dtype, device=device)
                local = lambda: _local(x, w_rows, y_shard, False)
                comm = lambda: dist.all_gather_into_tensor(gathered, y_shard)
                total = lambda: (
                    _local(x, w_rows, y_shard, False),
                    dist.all_gather_into_tensor(gathered, y_shard),
                )
                check_out = gathered.permute(1, 0, 2).reshape(t, h)
                check_ref = ref_full
            else:
                raise ValueError(f"unknown mode: {mode}")

            # ---- 正确性（只在第一次执行时检查）----
            total()
            stats = error_stats(check_out, check_ref)
            bound = atol + rtol * stats["ref_max_abs"]
            ok = stats["max_abs"] <= bound
            if rank == 0:
                status = "PASS" if ok else "FAIL"
                print(
                    f"[{mode}] {status} max_abs={stats['max_abs']:.3e} "
                    f"bound={bound:.3e} max_rel={stats['max_rel']:.3e}"
                )
            if not ok:
                raise RuntimeError(f"{mode} 正确性失败: max_abs={stats['max_abs']:.3e}")

            if cfg["check_only"]:
                all_metrics.append({"mode": mode, "skipped": False, "check": "PASS"})
                continue

            # ---- 计时（多轮取中位数）----
            local_rounds, comm_rounds, total_rounds, single_rounds = [], [], [], []
            total_detail: dict[str, float] = {}
            for ri in range(cfg["repeat"]):
                local_rounds.append(
                    bench_callable(local, warmup=cfg["warmup"], rep=cfg["rep"], device=device)["ms"]
                )
                comm_rounds.append(
                    bench_callable(comm, warmup=cfg["warmup"], rep=cfg["rep"], device=device)["ms"]
                )
                stat_total = bench_callable(total, warmup=cfg["warmup"], rep=cfg["rep"], device=device)
                total_rounds.append(stat_total["ms"])
                if ri == 0:
                    total_detail = {
                        k: float(stat_total[k])
                        for k in ("mean_ms", "min_ms", "max_ms", "std_ms", "inner_iters", "samples")
                    }
                single_rounds.append(
                    bench_callable(single_fn, warmup=cfg["warmup"], rep=cfg["rep"], device=device)["ms"]
                )

            local_ms = _median(local_rounds)
            comm_ms = _median(comm_rounds)
            total_ms = _median(total_rounds)
            single_ms = _median(single_rounds)
            speedup = single_ms / total_ms
            efficiency = speedup / world

            all_metrics.append(
                {
                    "mode": mode,
                    "skipped": False,
                    "dtype": dtype_name(out_dtype),
                    "local_ms": local_ms,
                    "comm_ms": comm_ms,
                    "total_ms": total_ms,
                    "tokens_per_s": t * 1e3 / total_ms,  # 每次迭代处理 T 个 token
                    "comm_ratio": comm_ms / total_ms,
                    "single_gpu_ms": single_ms,
                    "speedup": speedup,
                    "parallel_efficiency": efficiency,
                    "latency_detail_ms": total_detail,
                    "correctness": "PASS",
                    "rounds": {
                        "local": local_rounds,
                        "comm": comm_rounds,
                        "total": total_rounds,
                        "single": single_rounds,
                    },
                }
            )
            if rank == 0:
                print(
                    f"[{mode}] local={local_ms:.4f}ms comm={comm_ms:.4f}ms "
                    f"total={total_ms:.4f}ms single={single_ms:.4f}ms "
                    f"speedup={speedup:.2f}x eff={efficiency * 100:.1f}%"
                )
                print(
                    f"[{mode}] 延迟分布(首轮采样, ms): "
                    f"mean={total_detail['mean_ms']:.4f} min={total_detail['min_ms']:.4f} "
                    f"max={total_detail['max_ms']:.4f} std={total_detail['std_ms']:.4f}"
                )
            dist.barrier()

        gathered_metrics = [None] * world
        dist.all_gather_object(gathered_metrics, all_metrics)

        if rank == 0:
            print(f"\n===== 汇总 (backend={backend}, dtype={dtype_name(dtype)}, "
                  f"T={t}, H={h}, I={i}) =====")
            rows = []
            for m in gathered_metrics[0]:
                if m.get("skipped"):
                    rows.append([m["mode"], "skipped", "-", "-", "-", "-", "-", "-"])
                    continue
                if cfg["check_only"]:
                    rows.append([m["mode"], "PASS", "-", "-", "-", "-", "-", "-"])
                    continue
                rows.append(
                    [
                        m["mode"],
                        f"{m['local_ms']:.4f}",
                        f"{m['comm_ms']:.4f}",
                        f"{m['total_ms']:.4f}",
                        f"{m['comm_ratio'] * 100:.1f}%",
                        f"{m['tokens_per_s']:.0f}",
                        f"{m['speedup']:.2f}x",
                        f"{m['parallel_efficiency'] * 100:.1f}%",
                    ]
                )
            print_table(["mode", "local_ms", "comm_ms", "total_ms", "comm%", "tok/s", "speedup", "eff"], rows)

            if cfg["save"] and not cfg["check_only"]:
                save_result(
                    "task4_tp2",
                    {
                        "backend": backend,
                        "dtype": dtype_name(dtype),
                        "shape": {"T": t, "H": h, "I": i},
                        "world_size": world,
                        "chunks": cfg["chunks"],
                        "modes": gathered_metrics[0],
                    },
                )
    finally:
        dist.barrier()
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _simulate(cfg: dict) -> int:
    """单进程模拟两个 rank：只验证 K 切分/all-gather 的数值与逻辑（不做真实通信）。"""
    device = torch.device("cpu")
    dtype = torch.float32
    t, h, i = cfg["tokens"], cfg["hidden"], cfg["intermediate"]
    if i % 2 or h % 2:
        print("[error] simulate 模式要求 intermediate 与 hidden 为偶数")
        return 2

    x, w = _make_data(t, h, i, dtype, device)
    ref = x.double() @ w.double().t()
    half_i, half_h = i // 2, h // 2

    x0, x1 = x[:, :half_i], x[:, half_i:]
    w0, w1 = w[:, :half_i], w[:, half_i:]

    def k_split():
        return torch.mm(x0, w0.t()) + torch.mm(x1, w1.t())

    def column_split():
        y0 = torch.mm(x, w[:half_h].t())
        y1 = torch.mm(x, w[half_h:].t())
        return torch.cat([y0, y1], dim=1)

    single = lambda: torch.mm(x, w.t())
    print("[simulate] 仅验证数学等价性；CPU 上两个虚拟 rank 共享算力，不能代表双卡性能")
    failures = 0
    for name, fn in (("k_split_allreduce", k_split), ("column_split_allgather", column_split)):
        out = fn()
        stats = error_stats(out, ref)
        bound = cfg["atol"] + cfg["rtol"] * stats["ref_max_abs"]
        ok = stats["max_abs"] <= bound
        failures += int(not ok)
        stats_ms = bench_callable(fn, warmup=cfg["warmup"], rep=cfg["rep"], device=device)["ms"]
        single_ms = bench_callable(single, warmup=cfg["warmup"], rep=cfg["rep"], device=device)["ms"]
        print(
            f"[{name}] {'PASS' if ok else 'FAIL'} max_abs={stats['max_abs']:.3e} "
            f"bound={bound:.3e}  sim_ms={stats_ms:.4f} single_ms={single_ms:.4f}"
        )
    print("\n" + ("SIMULATION PASS" if failures == 0 else "SIMULATION FAIL"))
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="two-GPU split projection benchmark")
    p.add_argument("--nproc", type=int, default=2)
    p.add_argument("--backend", type=str, default="", choices=["", "nccl", "gloo"])
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--intermediate", type=int, default=14336)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--modes", type=str, nargs="+", default=["all"],
                     choices=["all", "allreduce_torch", "allreduce_triton",
                              "chunked_overlap", "p2p_reduce", "column_parallel"])
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--simulate", action="store_true",
                     help="单进程模拟两个 rank（CPU 逻辑校验，不做真实通信）")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl" and torch.cuda.device_count() < args.nproc:
        print(f"[warn] 需要 {args.nproc} 张 GPU，检测到 {torch.cuda.device_count()} 张，回退 gloo/CPU")
        backend = "gloo"

    dtype = args.dtype if backend == "nccl" else "fp32"
    if args.intermediate % 2 != 0 or args.hidden % args.nproc != 0:
        print("[error] intermediate 必须为偶数，hidden 必须能被 nproc 整除")
        return 2

    modes = (
        ["allreduce_torch", "allreduce_triton", "chunked_overlap", "p2p_reduce", "column_parallel"]
        if args.modes == ["all"]
        else args.modes
    )
    atol, rtol = (2e-2, 2e-2) if dtype in ("fp16", "bf16") else (1e-4, 1e-4)

    cfg = {
        "backend": backend,
        "simulate": args.simulate,
        "dtype": dtype,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "modes": modes,
        "chunks": max(1, args.chunks),
        "warmup": args.warmup,
        "rep": args.rep,
        "repeat": max(1, args.repeat),
        "check_only": args.check_only,
        "save": args.save,
        "atol": atol,
        "rtol": rtol,
    }

    if args.simulate or (backend == "gloo" and sys.platform == "win32"):
        if not args.simulate:
            print("[warn] Windows 的 gloo 后端不可用，自动切换为 --simulate 逻辑校验")
        cfg["dtype"] = "fp32"
        cfg["atol"], cfg["rtol"] = 1e-4, 1e-4
        return _simulate(cfg)

    # torchrun 已提供 rank/world_size 环境变量，直接进入 worker，避免再 mp.spawn 重复派生
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank == 0:
            print(f"[info] 检测到 torchrun：world_size={env_world}，使用 env:// 初始化")
        _worker(rank, env_world, cfg, "env://")
        return 0

    init_file = Path(tempfile.gettempdir()) / f"tp2_{uuid.uuid4().hex}"
    init_uri = init_file.as_uri()
    try:
        mp.spawn(_worker, args=(args.nproc, cfg, init_uri), nprocs=args.nproc, join=True)
    finally:
        try:
            init_file.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
