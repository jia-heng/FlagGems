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

"""Ascend mHC Post operator, using the TLE DSA surface.

Computes, with ``hc`` fixed at 4::

    out[t, i, d] = post_layer_mix[t, i] * x[t, d]
                 + sum_j comb_res_mix[t, j, i] * residual[t, j, d]

This operator is **DMA-issue-bound rather than bandwidth-bound**: every element
is touched exactly once, so there is no reuse to exploit, and device time tracks
the number of DMA instructions issued rather than the bytes moved. Tiling
decisions here are therefore ranked on DMA instruction count; a GB/s figure is
not a meaningful objective for this kernel.

There are two distinct performance regimes, with a separate kernel and tiling
rule for each. ``mhc_post`` below picks between them:

* **H >= 4096** -- cost is governed by the number of D-chunks, so ``BLOCK_D``
  is taken as wide as the UB budget allows (``_choose_block_d``) to minimize
  the chunk count. Kernel: ``_kernel_rows_pipeline``, grid ``(T,)``.
* **H <= 3072** -- cost is a per-token constant, not per byte. The ``(T,)``
  grid pays that constant T times where the vendor kernel pays it once per
  core, so tokens are batched ``BLOCK_T`` at a time across ``PIPE`` groups
  (``_choose_batch_t``). Kernel: ``_kernel_batch_t``, grid
  ``(ceil(T / BLOCK_T) / PIPE,)``.

The two rules are complementary and both ship: ``_choose_batch_t`` returns
``(0, 0)`` at H >= 4096 to decline, and ``_choose_block_d`` has no lever at
small H where it returns a single chunk.

Measured against ``aclnnMhcPost`` (fp16, device clock, L2 flushed); a ratio
above 1 is faster than the vendor kernel::

    T \\ H     1024   2048   4096   7168
    2048       0.97   0.72   1.04   1.27
    4096       0.86   0.67   1.27   1.34
    8192       0.90   0.72   1.29   1.34

H=2048 at ~0.72 is a backend ceiling rather than a tuning gap: the kernel is
vector-compute bound there, and the vendor's advantage is the fused
scalar-vector multiply-add of AscendC ``Axpy`` (~20*H element-ops per token
against ~36*H unfused here). ``tl.math.fma`` compiles but emits the same code
as an explicit multiply plus add on this backend, so the fusion is not
reachable from Triton today. Accumulating in fp16 would recover it, but the
vendor kernel accumulates in fp32 and matching its accuracy takes priority.

Only ``hc == 4`` is handled here. Any other ``hc``, a hidden size that is not a
multiple of 128, a non-contiguous or oddly-typed input, or a T beyond the grid
cap falls back to the generic implementation in ``flag_gems.fused.mhc``.
"""

import logging
import os

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle as tle

    _HAS_DSA = hasattr(tle, "dsa")
except ImportError:  # pragma: no cover - TLE absent in some toolchains
    tle = None
    _HAS_DSA = False

from flag_gems.fused.mhc.mhc_post import mhc_post as _mhc_post_generic

logger = logging.getLogger(__name__)

# ``USE_CUSTOM_COMPILE_OPT`` pins num_stages to 1 (it implies
# --enable-auto-multi-buffer=False), which disables the double buffering both
# kernels below rely on and costs a flat ~25%. Warn once rather than unset it:
# the variable is process-global and other operators in the same process may be
# depending on it.
if os.environ.get("USE_CUSTOM_COMPILE_OPT"):
    logger.warning(
        "USE_CUSTOM_COMPILE_OPT is set; it pins num_stages=1 and disables the "
        "auto-multi-buffer that mhc_post's pipelining depends on, costing "
        "roughly 25%% on this operator. Unset it to recover full performance."
    )


# ---------------------------------------------------------------------------
# Tiling rules
# ---------------------------------------------------------------------------

# UB bytes per element of BLOCK_D for the pipelined kernel. **Measured, not
# derived**: the compiler reports its own requirement when a tile does not fit,
# and this is that number read back at D=7168 fp16 (BLOCK_D=4608 -> 1696512
# bits = 212064 B -> 46.02 B/elem; BLOCK_D=5120 -> 1884928 bits, same ratio).
# For reference AscendC spends 24 B/element, keeping one fp32 accumulator live
# instead of a [4, BLOCK_D] broadcast plus 5 fp32 inputs, which is why its
# dInner reaches 8192 and ours cannot.
_UB_BYTES_PER_ELEM = 46
_UB_BUDGET_BYTES = 192 * 1024
# 512 B burst floor for fp16, matching ALIGN_SIZE_512B in the AscendC tiling.
_MIN_BLOCK_D = 256
# DataCopyPad wants 32 B alignment; 16 fp16 elements is the granularity the
# AscendC tiling rounds dInner to.
_BLOCK_D_ALIGN = 16

# UB cost of the batched-token kernel, also read off the compiler's overflow
# messages rather than hand-derived. Two terms, because the staged input scales
# with BLOCK_T while the fp32 working set of the inner token loop does not:
#
#     bytes = _UB_BT_PER_ELEM * BLOCK_T * D + _UB_BT_FIXED_PER_D * D
#
# Fitted against the reported requirement (bits -> bytes):
#
#     D=1024 BT=16 -> 206080 B    10*16*1024 + 40*1024 = 204800
#     D=2048 BT=8  -> 246400 B    10*8*2048  + 40*2048 = 245760
#     D=4096 BT=2  -> 229536 B    10*2*4096  + 40*4096 = 245760
#
# 10 B/element is the staged input (x at 8 B per 4-row element plus h_out at
# 2 B); the ~40 B/element floor is the per-token fp32 temporaries. That floor is
# why D >= 4096 cannot batch at all -- 40*4096 = 160 KB of the budget is gone
# before the first token is staged -- which is the right outcome, since large D
# already leads and does not need batching.
_UB_BT_PER_ELEM = 10
_UB_BT_FIXED_PER_D = 40
# The budget the compiler enforces for this kernel: 1572864 bits. Note this is
# 192 KiB expressed as 1536 Kib, so the usable figure is 196608 B.
_UB_BT_BUDGET_BYTES = 1572864 // 8
# BLOCK_T is capped well below the UB bound, for two reasons that both say 4.
# The inner token loop is a fully unrolled static_range, so BLOCK_T is also the
# code-size multiplier, while the DMA it saves is already down to 0.25 per token
# at 4. More importantly BLOCK_T and PIPE compete for the same UB, and PIPE is
# the better buy: at D=1024 T=8192, BLOCK_T=8/PIPE=4 measured 0.2622 ms against
# 0.2263 for BLOCK_T=4/PIPE=4 -- 14% worse for spending the budget on batch
# width instead of pipeline depth.
_MAX_BLOCK_T = 4
# coreDim cap on the launch grid.
_MAX_GRID = 65535


def _num_vector_cores(default: int = 40) -> int:
    """Vector core count of the current device, or ``default`` if unavailable."""
    try:
        dev = torch.npu.current_device()
        return int(torch.npu.get_device_limit(dev)["vector_core_num"])
    except (AttributeError, KeyError, TypeError, RuntimeError):
        return default


def _ub_block_d_cap() -> int:
    """Largest BLOCK_D that still compiles with the pipeline intact."""
    cap = _UB_BUDGET_BYTES // _UB_BYTES_PER_ELEM
    return (cap // _BLOCK_D_ALIGN) * _BLOCK_D_ALIGN  # 4272 at 192 KB / 46 B


def _choose_block_d(D: int) -> int:
    """Fewest D-chunks the UB budget allows, rounded up to 16 elements.

    Every chunk costs 2 loads + 4 stores regardless of width, so the tile wants
    to be as wide as UB permits. Measured at T=2048 fp16, sweeping chunk count::

        D=7168   nblk 8/6/5/4/3/2   -> 0.446 0.450 0.407 0.403 0.369 0.378
        D=4096   nblk 8/6/5/4/3/2/1 -> 0.349 0.341 0.268 0.274 0.260 0.252 0.219
        D=2560   nblk 8/6/5/4/3/2/1 -> 0.326 0.309 0.241 0.221 0.206 0.189 0.167

    Monotone in chunk count on three of four; D=7168 is within 2.4% of its own
    optimum, not enough structure to justify special-casing. Rounding to
    ``ceil(D / nblk)`` rather than a power of two keeps the tail waste at one
    alignment unit, which matters because a chunk always computes BLOCK_D
    elements and is not masked down to the tail.
    """
    cap = _ub_block_d_cap()
    nblk = max(1, triton.cdiv(D, cap))
    block = triton.cdiv(triton.cdiv(D, nblk), _BLOCK_D_ALIGN) * _BLOCK_D_ALIGN
    return max(_MIN_BLOCK_D if D >= _MIN_BLOCK_D else D, min(block, cap))


def _choose_batch_t(D: int, T: int, elem_bytes: int = 2):
    """Pick ``(BLOCK_T, PIPE)`` for the batched kernel, or ``(0, 0)`` to decline.

    ``BLOCK_T`` is how many tokens are staged per DMA group, bounded by UB.
    ``PIPE`` is how many such groups one program walks under ``num_stages=2``;
    it exists to give the pipeline something to overlap, which at small D the
    D-chunk loop cannot do because there is only one chunk.

    Measured, fp16, device clock, L2 flushed, T=8192::

        D=1024  BT/PIPE  1/1   2/1   4/1   8/1   2/2   4/2   4/4   2/4
                dev ms  0.603 0.308 0.252 0.251 0.245 0.237 0.226 0.229
        D=2048  BT/PIPE  1/1   2/1   4/1   2/2   4/2   4/4   2/4
                dev ms  0.605 0.435 0.422 0.409 0.403 0.386 0.382

    PIPE's cap of 4 is a measured optimum, swept past it at BLOCK_T=4::

        D=1024  PIPE 1/2/4/8/16 -> 0.2518 0.2374 0.2266 0.2349 0.2633 ms
        D=2048  PIPE 1/2/4/8/16 -> 0.4204 0.4016 0.3820 0.3969 0.4445 ms

    Both turn at 4 and are worse by 16, where the grid has stopped covering the
    vector cores evenly.
    """
    if D <= 0 or T < 2:
        return 0, 0
    # Scale the fitted fp16 constants if the element width differs.
    scale = elem_bytes / 2
    fixed = int(_UB_BT_FIXED_PER_D * D)
    per_tok = max(1, int(_UB_BT_PER_ELEM * scale * D))
    block_t = (_UB_BT_BUDGET_BYTES - fixed) // per_tok
    block_t = min(block_t, T, _MAX_BLOCK_T)
    if block_t < 2:
        return 0, 0
    # Largest power of two that fits: the inner loop is fully unrolled, so a
    # non-power-of-two BLOCK_T only adds code without reducing DMA count.
    block_t = 1 << (block_t.bit_length() - 1)
    # Keep enough groups to fill the vector cores before growing PIPE.
    cores = _num_vector_cores()
    pipe = 1
    while pipe < 4 and T // (block_t * pipe * 2) >= cores:
        pipe *= 2
    return block_t, pipe


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _kernel_rows_pipeline(
    x_ptr,  # residual  (T, 4, D) bf16/fp16
    h_res_ptr,  # comb_res_mix (T, 4, 4) fp32
    h_out_ptr,  # x         (T, D)    bf16/fp16
    h_post_ptr,  # post_layer_mix (T, 4) fp32
    out_ptr,  # out       (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    """One program per token, walking D in pipelined chunks. For large D."""
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # Coefficients are constant across D-chunks, so they are staged once.
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32
    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # num_stages=2 double-buffers the staged tile: chunk N+1 loads while chunk N
    # computes.
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        Y = (
            tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)
            + tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)
            + tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)
            + tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)
            + tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)
        )  # [4, BLOCK_D]

        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


@triton.jit
def _kernel_batch_t(
    x_ptr,  # residual  (T, 4, D) bf16/fp16
    h_res_ptr,  # comb_res_mix (T, 4, 4) fp32
    h_out_ptr,  # x         (T, D)    bf16/fp16
    h_post_ptr,  # post_layer_mix (T, 4) fp32
    out_ptr,  # out       (T, 4, D) bf16/fp16
    T,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PIPE: tl.constexpr,
    EXACT: tl.constexpr,  # T % (BLOCK_T * PIPE) == 0, so no group is partial
):
    """One program owns BLOCK_T tokens x PIPE groups. For small D.

    The ``(T,)``-grid kernel above pays a per-token fixed cost T times. At small
    D that cost, not bandwidth, is the whole story: identical traffic measured
    2.76x apart (D=1024 T=8192 at 0.6035 ms vs D=4096 T=2048 at 0.2185 ms, both
    144 MB), with per-token cost flat across small D (73.8 ns at D=1024, 75.7 at
    D=2048) while the bytes double. The vendor kernel pays it once per core and
    loops tokens inside; this does the same.

    Batching removes two per-token components. First, DMA instruction count: at
    D=1024 a token costs 8 DMAs, two of which are the 16 B and 64 B coefficient
    loads -- 25% of the issue count for 0.7% of the bytes. Because ``x[t]`` is a
    contiguous ``(4, D)`` block, a run of tokens is one contiguous region, so
    the x and h_out loads stay at one DMA each per group however large BLOCK_T
    is. Second, lost pipelining: at small D there is a single D-chunk, so the
    other kernel's pipeline loop has nothing to overlap, while the PIPE loop
    here restores a multi-iteration loop independent of D.

    Stores stay per row. Computing all BLOCK_T tokens into one staged buffer and
    copying out once measured 34 ms against 0.25 ms -- UB->GM writes with a
    non-trivial geometry get scalarized. Only the *read* side of such a copy
    lowers to an efficient DataCopyPad.
    """
    pid = tl.program_id(0)
    x_dt = x_ptr.dtype.element_ty

    XN: tl.constexpr = BLOCK_T * 4 * D
    HN: tl.constexpr = BLOCK_T * D

    hpost_ub = tle.dsa.alloc(
        [BLOCK_T * 4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    hres_ub = tle.dsa.alloc(
        [BLOCK_T * 16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    x_ub = tle.dsa.alloc([XN], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    ho_ub = tle.dsa.alloc([HN], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    for g in tle.dsa.pipeline(0, PIPE, 1, num_stages=2):
        t0 = (pid * PIPE + g) * BLOCK_T
        n_tok = BLOCK_T if EXACT else tl.minimum(BLOCK_T, T - t0)
        if EXACT or n_tok > 0:
            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(
                    h_post_ptr + t0 * 4 + tl.arange(0, BLOCK_T * 4),
                    hpost_ub,
                    [n_tok * 4],
                )
                tle.dsa.copy(
                    h_res_ptr + t0 * 16 + tl.arange(0, BLOCK_T * 16),
                    hres_ub,
                    [n_tok * 16],
                )
            tle.dsa.copy(x_ptr + t0 * 4 * D + tl.arange(0, XN), x_ub, [n_tok * 4 * D])
            tle.dsa.copy(h_out_ptr + t0 * D + tl.arange(0, HN), ho_ub, [n_tok * D])

            hp_vec = tle.dsa.to_tensor(hpost_ub)
            hr_vec = tle.dsa.to_tensor(hres_ub)
            x_f = tle.dsa.to_tensor(x_ub).to(tl.float32)
            ho_f = tle.dsa.to_tensor(ho_ub).to(tl.float32)

            for tt in tl.static_range(BLOCK_T):
                if EXACT or tt < n_tok:
                    hp = tl.reshape(
                        tle.dsa.extract_slice(hp_vec, (tt * 4,), (4,), (1,)), [4]
                    )
                    hr0 = tl.reshape(
                        tle.dsa.extract_slice(hr_vec, (tt * 16 + 0,), (4,), (1,)), [4]
                    )
                    hr1 = tl.reshape(
                        tle.dsa.extract_slice(hr_vec, (tt * 16 + 4,), (4,), (1,)), [4]
                    )
                    hr2 = tl.reshape(
                        tle.dsa.extract_slice(hr_vec, (tt * 16 + 8,), (4,), (1,)), [4]
                    )
                    hr3 = tl.reshape(
                        tle.dsa.extract_slice(hr_vec, (tt * 16 + 12,), (4,), (1,)), [4]
                    )

                    xb = tt * 4 * D
                    x0 = tl.reshape(
                        tle.dsa.extract_slice(x_f, (xb + 0 * D,), (D,), (1,)), [D]
                    )
                    x1 = tl.reshape(
                        tle.dsa.extract_slice(x_f, (xb + 1 * D,), (D,), (1,)), [D]
                    )
                    x2 = tl.reshape(
                        tle.dsa.extract_slice(x_f, (xb + 2 * D,), (D,), (1,)), [D]
                    )
                    x3 = tl.reshape(
                        tle.dsa.extract_slice(x_f, (xb + 3 * D,), (D,), (1,)), [D]
                    )
                    ho = tl.reshape(
                        tle.dsa.extract_slice(ho_f, (tt * D,), (D,), (1,)), [D]
                    )

                    Y = (
                        tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)
                        + tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)
                        + tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)
                        + tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)
                        + tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)
                    )
                    Yc = Y.to(x_dt)

                    d_off = tl.arange(0, D)
                    base = (t0 + tt) * 4 * D
                    y0 = tl.reshape(
                        tle.dsa.extract_slice(Yc, (0, 0), (1, D), (1, 1)), [D]
                    )
                    y1 = tl.reshape(
                        tle.dsa.extract_slice(Yc, (1, 0), (1, D), (1, 1)), [D]
                    )
                    y2 = tl.reshape(
                        tle.dsa.extract_slice(Yc, (2, 0), (1, D), (1, 1)), [D]
                    )
                    y3 = tl.reshape(
                        tle.dsa.extract_slice(Yc, (3, 0), (1, D), (1, 1)), [D]
                    )
                    tl.store(out_ptr + base + 0 * D + d_off, y0)
                    tl.store(out_ptr + base + 1 * D + d_off, y1)
                    tl.store(out_ptr + base + 2 * D + d_off, y2)
                    tl.store(out_ptr + base + 3 * D + d_off, y3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _tle_eligible(x, residual, post_layer_mix, comb_res_mix, hc, h, n):
    """Whether the TLE path can handle this call, and why not when it cannot."""
    if not _HAS_DSA:
        return False, "TLE DSA surface unavailable"
    if hc != 4:
        return False, f"hc={hc} (TLE path is hc=4 only)"
    if h % 128 != 0:
        return False, f"H={h} is not a multiple of 128"
    if h <= 0 or n <= 0:
        return False, f"empty input (N={n}, H={h})"
    if residual.dtype not in (torch.float16, torch.bfloat16):
        return False, f"residual dtype {residual.dtype}"
    if x.dtype != residual.dtype:
        return False, f"x dtype {x.dtype} != residual dtype {residual.dtype}"
    # The coefficient loads below read fp32 directly from GM.
    if post_layer_mix.dtype != torch.float32 or comb_res_mix.dtype != torch.float32:
        return False, "post_layer_mix/comb_res_mix must be float32"
    return True, None


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post-processing operator, Ascend TLE implementation.

    Args:
        x: (N, H) bf16/fp16 -- layer output.
        residual: (N, hc_mult, H) bf16/fp16 -- multi-head residual.
        post_layer_mix: (N, hc_mult, 1) or (N, hc_mult) fp32 -- scale for x.
        comb_res_mix: (N, hc_mult, hc_mult) fp32 -- combination matrix.

    Returns:
        out: (N, hc_mult, H), same dtype as ``residual``.

    Falls back to the generic implementation for any shape or dtype the TLE
    path does not cover; see ``_tle_eligible``.
    """
    logger.debug(
        "GEMS ASCEND MHC_POST FORWARD, x=%s, residual=%s, post_layer_mix=%s, "
        "comb_res_mix=%s",
        x.shape,
        residual.shape,
        post_layer_mix.shape,
        comb_res_mix.shape,
    )

    n, hc, h = residual.shape
    assert x.shape == (n, h)
    assert post_layer_mix.shape in ((n, hc, 1), (n, hc))
    assert comb_res_mix.shape == (n, hc, hc)

    ok, reason = _tle_eligible(x, residual, post_layer_mix, comb_res_mix, hc, h, n)
    if not ok:
        logger.debug("mhc_post: falling back to generic kernel (%s)", reason)
        return _mhc_post_generic(x, residual, post_layer_mix, comb_res_mix)

    xf = residual.contiguous()  # (N, 4, H)
    hout = x.contiguous()  # (N, H)
    hpost = post_layer_mix.reshape(n, hc).contiguous()  # (N, 4) fp32
    hres = comb_res_mix.contiguous()  # (N, 4, 4) fp32

    block_t, pipe = _choose_batch_t(h, n, residual.element_size())

    if block_t >= 2:
        # Small-H regime: batch tokens. The grid divides by BLOCK_T*PIPE, so it
        # clears the coreDim cap on its own.
        groups = triton.cdiv(n, block_t)
        grid = (triton.cdiv(groups, pipe),)
        out = torch.empty_like(xf)
        _kernel_batch_t[grid](
            xf,
            hres,
            hout,
            hpost,
            out,
            n,
            D=h,
            BLOCK_T=block_t,
            PIPE=pipe,
            EXACT=(n % (block_t * pipe) == 0),
        )
        return out

    # Large-H regime: one program per token. Here the grid is T itself, so the
    # coreDim cap binds and BLOCK_D cannot relieve it.
    if n > _MAX_GRID:
        logger.debug(
            "mhc_post: falling back to generic kernel (N=%d exceeds the %d "
            "coreDim cap on the unbatched path)",
            n,
            _MAX_GRID,
        )
        return _mhc_post_generic(x, residual, post_layer_mix, comb_res_mix)

    block_d = _choose_block_d(h)
    out = torch.empty_like(xf)
    _kernel_rows_pipeline[(n,)](
        xf,
        hres,
        hout,
        hpost,
        out,
        D=h,
        BLOCK_D=block_d,
        NUM_D_BLOCKS=triton.cdiv(h, block_d),
    )
    return out
