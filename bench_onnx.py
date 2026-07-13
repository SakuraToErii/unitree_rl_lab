#!/usr/bin/env python3
"""Benchmark an exported ONNX policy on the deployment computer.

This measures ONNX Runtime inference latency for the exact exported model. Run
the script on the computer and under the system load used for deployment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


DTYPE_MAP: dict[str, np.dtype] = {
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(bool)": np.dtype(np.bool_),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def headroom_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in the interval (0, 1]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure p50/p95/p99 inference latency of an exported ONNX policy."
    )
    parser.add_argument("model", type=Path, help="Path to policy.onnx")
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=500,
        help="Warm-up inference count (default: 500).",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=10_000,
        help="Measured inference count (default: 10000).",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider (default: CPUExecutionProvider).",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=positive_int,
        default=None,
        help="Override ONNX Runtime intra-op thread count.",
    )
    parser.add_argument(
        "--headroom",
        type=headroom_ratio,
        default=0.8,
        help="Fraction of each control period available to inference (default: 0.8).",
    )
    return parser.parse_args()


def concrete_shape(shape: list[int | str | None]) -> list[int]:
    """Resolve dynamic dimensions to one for a single-policy inference batch."""
    return [dimension if isinstance(dimension, int) and dimension > 0 else 1 for dimension in shape]


def create_session(args: argparse.Namespace) -> ort.InferenceSession:
    available_providers = ort.get_available_providers()
    if args.provider not in available_providers:
        choices = ", ".join(available_providers)
        raise RuntimeError(f"Provider {args.provider!r} is unavailable. Available providers: {choices}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if args.intra_op_threads is not None:
        options.intra_op_num_threads = args.intra_op_threads

    return ort.InferenceSession(
        args.model.as_posix(),
        sess_options=options,
        providers=[args.provider],
    )


def create_inputs(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    print("Inputs:")
    for model_input in session.get_inputs():
        dtype = DTYPE_MAP.get(model_input.type)
        if dtype is None:
            supported = ", ".join(sorted(DTYPE_MAP))
            raise TypeError(f"Unsupported ONNX input type {model_input.type!r}. Supported types: {supported}")

        shape = concrete_shape(model_input.shape)
        feeds[model_input.name] = np.zeros(shape, dtype=dtype)
        print(f"  {model_input.name}: shape={shape}, type={model_input.type}")

    print("Outputs:")
    for model_output in session.get_outputs():
        print(f"  {model_output.name}: shape={model_output.shape}, type={model_output.type}")
    return feeds


def benchmark(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    warmup: int,
    iterations: int,
) -> np.ndarray:
    print(f"\nWarming up for {warmup} iterations...")
    for _ in range(warmup):
        session.run(None, feeds)

    print(f"Measuring {iterations} iterations...")
    latency_ms = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start_ns = time.perf_counter_ns()
        session.run(None, feeds)
        end_ns = time.perf_counter_ns()
        latency_ms[index] = (end_ns - start_ns) / 1.0e6
    return latency_ms


def print_results(latency_ms: np.ndarray, headroom: float) -> None:
    p50, p95, p99 = np.percentile(latency_ms, [50, 95, 99])
    mean = float(np.mean(latency_ms))
    maximum = float(np.max(latency_ms))
    safe_rate_hz = 1000.0 * headroom / p99

    print("\nLatency:")
    print(f"  mean: {mean:9.3f} ms")
    print(f"  p50 : {p50:9.3f} ms")
    print(f"  p95 : {p95:9.3f} ms")
    print(f"  p99 : {p99:9.3f} ms")
    print(f"  max : {maximum:9.3f} ms")
    print(f"  sequential throughput: {1000.0 / mean:9.1f} Hz")
    print(f"  p99 rate with {headroom:.0%} period budget: {safe_rate_hz:9.1f} Hz")

    print("\nControl-rate inference budget:")
    for rate_hz in (50, 100, 200, 500):
        period_ms = 1000.0 / rate_hz
        budget_ms = period_ms * headroom
        status = "PASS" if p99 <= budget_ms else "OVER"
        print(
            f"  {rate_hz:3d} Hz: period={period_ms:6.2f} ms, "
            f"budget={budget_ms:6.2f} ms, p99={p99:6.3f} ms  {status}"
        )


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    args.model = model_path

    print(f"Model: {model_path}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Provider: {args.provider}")
    print(f"Available providers: {ort.get_available_providers()}")

    session = create_session(args)
    feeds = create_inputs(session)
    latency_ms = benchmark(session, feeds, args.warmup, args.iterations)
    print_results(latency_ms, args.headroom)


if __name__ == "__main__":
    main()
