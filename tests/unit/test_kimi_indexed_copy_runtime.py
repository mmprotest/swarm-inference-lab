from __future__ import annotations

import ctypes

import pytest

from swarm_inference.execution.kimi_cuda_runtime import KimiCudaError, _CudaRuntime


def test_indexed_copy_rows_calls_optional_native_export() -> None:
    observed: list[tuple[object, ...]] = []
    runtime = object.__new__(_CudaRuntime)
    runtime.device = 3

    def native(*args: object) -> int:
        observed.append(args)
        return 1

    runtime.indexed_copy_rows_function = native
    output = ctypes.c_void_p(11)
    source = ctypes.c_void_p(12)
    indices = ctypes.c_void_p(13)

    runtime.execute_indexed_copy_rows(
        output,
        source,
        indices,
        rows=128,
        dimension=3584,
    )

    assert observed == [(3, output, source, indices, 128, 3584)]


def test_indexed_copy_rows_falls_closed_for_legacy_binary() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.device = 0
    runtime.indexed_copy_rows_function = None

    with pytest.raises(KimiCudaError, match="no indexed row-copy export"):
        runtime.execute_indexed_copy_rows(
            ctypes.c_void_p(1),
            ctypes.c_void_p(2),
            ctypes.c_void_p(3),
            rows=1,
            dimension=1,
        )


def test_indexed_copy_rows_propagates_native_failure() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.device = 0
    runtime.indexed_copy_rows_function = lambda *_args: 0

    with pytest.raises(KimiCudaError, match="indexed row copy failed"):
        runtime.execute_indexed_copy_rows(
            ctypes.c_void_p(1),
            ctypes.c_void_p(2),
            ctypes.c_void_p(3),
            rows=1,
            dimension=1,
        )


def test_triangular_mla_calls_optional_native_export() -> None:
    observed: list[tuple[object, ...]] = []
    runtime = object.__new__(_CudaRuntime)

    def native(*args: object) -> int:
        observed.append(args)
        return 1

    runtime.mla_absorb_triangular_function = native
    pointers = tuple(ctypes.c_void_p(value) for value in range(11, 16))
    runtime.execute_mla_absorb_triangular(
        *pointers,
        batch=8,
        heads=64,
        query_nope=128,
        query_rope=64,
        value_dimension=128,
        kv_lora=512,
        final_context_length=8200,
        attention_scale=0.125,
    )

    assert len(observed) == 1
    assert observed[0][:5] == pointers
    assert observed[0][5:12] == (8, 64, 128, 64, 128, 512, 8200)
    assert float(observed[0][12].value) == pytest.approx(0.125)


def test_triangular_mla_falls_closed_for_legacy_binary() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.mla_absorb_triangular_function = None

    with pytest.raises(KimiCudaError, match="no triangular MLA export"):
        runtime.execute_mla_absorb_triangular(
            *(ctypes.c_void_p(value) for value in range(1, 6)),
            batch=2,
            heads=1,
            query_nope=1,
            query_rope=1,
            value_dimension=1,
            kv_lora=1,
            final_context_length=2,
            attention_scale=1.0,
        )


def test_triangular_mla_propagates_native_failure() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.mla_absorb_triangular_function = lambda *_args: 0

    with pytest.raises(KimiCudaError, match="triangular MLA attention rejected"):
        runtime.execute_mla_absorb_triangular(
            *(ctypes.c_void_p(value) for value in range(1, 6)),
            batch=2,
            heads=1,
            query_nope=1,
            query_rope=1,
            value_dimension=1,
            kv_lora=1,
            final_context_length=2,
            attention_scale=1.0,
        )


def test_dcp_mla_calls_optional_native_export() -> None:
    observed: list[tuple[object, ...]] = []
    runtime = object.__new__(_CudaRuntime)

    def native(*args: object) -> int:
        observed.append(args)
        return 1

    runtime.mla_absorb_dcp_function = native
    pointers = tuple(ctypes.c_void_p(value) for value in range(21, 26))
    runtime.execute_mla_absorb_dcp(
        *pointers,
        batch=8,
        degree=4,
        heads=64,
        query_nope=128,
        query_rope=64,
        value_dimension=128,
        kv_lora=512,
        final_context_length=8200,
        attention_scale=0.125,
    )

    assert len(observed) == 1
    assert observed[0][:5] == pointers
    assert observed[0][5:13] == (8, 4, 64, 128, 64, 128, 512, 8200)
    assert float(observed[0][13].value) == pytest.approx(0.125)


def test_dcp_mla_falls_closed_for_legacy_binary() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.mla_absorb_dcp_function = None

    with pytest.raises(KimiCudaError, match="no DCP MLA export"):
        runtime.execute_mla_absorb_dcp(
            *(ctypes.c_void_p(value) for value in range(1, 6)),
            batch=2,
            degree=2,
            heads=1,
            query_nope=1,
            query_rope=1,
            value_dimension=1,
            kv_lora=1,
            final_context_length=2,
            attention_scale=1.0,
        )


def test_kda_short_window_calls_optional_native_export() -> None:
    observed: list[tuple[object, ...]] = []
    runtime = object.__new__(_CudaRuntime)
    runtime.device = 2

    def native(*args: object) -> int:
        observed.append(args)
        return 1

    runtime.kda_short_window_function = native
    pointers = tuple(ctypes.c_void_p(value) for value in range(1, 18))
    runtime.execute_kda_short_window(
        *pointers,
        rows=8,
        heads=96,
        head_dimension=128,
        convolution_width=4,
        gate_lower_bound=-5.0,
        epsilon=1e-5,
    )

    assert len(observed) == 1
    assert observed[0][0] == 2
    assert observed[0][1:18] == pointers
    assert observed[0][18:22] == (8, 96, 128, 4)


def test_kda_short_window_fails_closed_for_legacy_binary() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.device = 0
    runtime.kda_short_window_function = None

    with pytest.raises(KimiCudaError, match="absent from the native runtime"):
        runtime.execute_kda_short_window(
            *(ctypes.c_void_p(value) for value in range(1, 18)),
            rows=1,
        )


def test_kda_short_window_propagates_native_failure_without_serial_fallback() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.device = 0
    runtime.kda_short_window_function = lambda *_args: 0

    with pytest.raises(KimiCudaError, match="short-window core rejected"):
        runtime.execute_kda_short_window(
            *(ctypes.c_void_p(value) for value in range(1, 18)),
            rows=16,
        )
