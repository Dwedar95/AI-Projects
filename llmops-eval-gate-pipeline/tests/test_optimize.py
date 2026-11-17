import onnx
import pytest

from llmops_gate.optimize import format_report, run_benchmark, time_calls


@pytest.fixture(scope="module")
def report_and_dir(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("bench")
    return run_benchmark(hidden=(64,), repeats=50, workdir=workdir), workdir


def test_benchmark_measures_three_variants(report_and_dir):
    report, _ = report_and_dir
    assert [v.name for v in report.variants] == ["sklearn-mlp", "onnx-float32", "onnx-int8-dynamic"]
    assert report.n_train + report.n_test == 1797
    for variant in report.variants:
        assert variant.accuracy > 0.9
        assert 0 < variant.single_p50_ms <= variant.single_p95_ms
        assert variant.batch_rows_per_s > 0


def test_onnx_export_is_faithful_and_quantisation_stays_close(report_and_dir):
    report, _ = report_and_dir
    sklearn_variant, float32, int8 = report.variants
    assert float32.accuracy == sklearn_variant.accuracy
    assert float32.agreement_with_float32 == 1.0 and sklearn_variant.agreement_with_float32 == 1.0
    assert int8.agreement_with_float32 > 0.97
    assert abs(int8.accuracy - float32.accuracy) < 0.03


def test_quantised_model_is_smaller_and_uses_integer_matmul(report_and_dir):
    report, workdir = report_and_dir
    assert report.variants[2].size_kb < 0.6 * report.variants[1].size_kb
    ops = {n.op_type for n in onnx.load(str(workdir / "digits_mlp_int8.onnx")).graph.node}
    float_ops = {n.op_type for n in onnx.load(str(workdir / "digits_mlp_float32.onnx")).graph.node}
    assert "MatMulInteger" in ops and "MatMulInteger" not in float_ops


def test_report_table_and_timer(report_and_dir):
    table = format_report(report_and_dir[0])
    assert "onnx-int8-dynamic" in table and "n/a" in table and "rows/s" in table
    durations = time_calls(lambda: sum(range(100)), repeats=10, warmup=2)
    assert durations.shape == (10,) and (durations > 0).all()
