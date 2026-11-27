# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark npu_dynamic_mx_quant against torch_npu.npu_dynamic_mx_quant.

Run on an Ascend950DT device::

    source /data/jianheng/export.sh
    python tests/bench_npu_dynamic_mx_quant.py
    python tests/bench_npu_dynamic_mx_quant.py --dtype fp16 --scale-alg 1
    python tests/bench_npu_dynamic_mx_quant.py --warmup 10 --iterations 20

The reported latency is the median NPU Event time over ``iterations`` rounds,
each round calling the op 10 times back-to-back.  ``DynamicMxQuantPlan`` is
pre-built before measurement so Triton compilation is excluded.  The vendor
path includes its ~16 us aclnn host dispatch in eager mode.
"""

import argparse
import importlib.util
import pathlib
import statistics

import torch
import torch_npu  # noqa: F401

# ---------------------------------------------------------------------------
# Import kernel module directly (flag_gems init may be broken on this machine)
# ---------------------------------------------------------------------------
_MOD_PATH = (
    pathlib.Path(__file__).parent.parent
    / "src/flag_gems/runtime/backend/_ascend/a5/ops/npu_dynamic_mx_quant.py"
)
_spec = importlib.util.spec_from_file_location("_npu_dmq", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DynamicMxQuantPlan = _mod.DynamicMxQuantPlan
should_use_triton = _mod.should_use_triton

# ---------------------------------------------------------------------------
# Shapes: (M, K) covering decode and prefill regimes for common LLM sizes
# ---------------------------------------------------------------------------
_SHAPES = [
    # decode
    (1, 512),
    (1, 7168),
    (4, 7168),
    (16, 512),
    (16, 7168),
    (64, 7168),
    (256, 512),
    (256, 7168),
    # prefill
    (1024, 7168),
    (4096, 512),
    (4096, 7168),
]

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_DST_MAP = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


def _time_us(fn, warmup: int, iterations: int) -> float:
    """Median NPU Event time in microseconds (10 calls per sample)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 100)  # ms*10 -> us
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description="bench npu_dynamic_mx_quant")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="input dtype",
    )
    parser.add_argument(
        "--dst-type",
        choices=["e4m3", "e5m2"],
        default="e4m3",
        help="output fp8 dtype",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--scale-alg", type=int, choices=[0, 1], default=0, help="0=FLOOR 1=CEIL"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    dst_type = _DST_MAP[args.dst_type]
    B = args.block_size

    header = (
        f"{'M':>6} {'K':>6}  {'rows':>4}  "
        f"{'ours(us)':>10}  {'vendor(us)':>10}  {'speedup':>8}  note"
    )
    print(header)
    print("-" * len(header))

    for M, K in _SHAPES:
        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=dtype, device="npu")

        plan = DynamicMxQuantPlan(
            x,
            dst_type=dst_type,
            block_size=B,
            axis=-1,
            round_mode="rint",
            scale_alg=args.scale_alg,
        )
        y_out = torch.empty(M, K, dtype=dst_type, device="npu")
        s_out = torch.zeros(plan.M, plan.nsp, dtype=torch.uint8, device="npu")

        def ours():
            plan(x, out=y_out, scale=s_out)

        def vendor():
            torch_npu.npu_dynamic_mx_quant(
                x,
                axis=-1,
                round_mode="rint",
                dst_type=dst_type,
                block_size=B,
                scale_alg=args.scale_alg,
            )

        ours_us = _time_us(ours, args.warmup, args.iterations)
        vendor_us = _time_us(vendor, args.warmup, args.iterations)
        speedup = vendor_us / ours_us
        note = f"rows={plan.rows}"
        if not should_use_triton(M, K, dtype, B):
            note += " [vendor recommended]"

        print(
            f"{M:>6} {K:>6}  {plan.rows:>4}  "
            f"{ours_us:>10.2f}  {vendor_us:>10.2f}  {speedup:>8.3f}x  {note}",
            flush=True,
        )


if __name__ == "__main__":
    main()
