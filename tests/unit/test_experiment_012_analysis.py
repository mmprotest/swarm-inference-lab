from __future__ import annotations

from swarm_inference.experiments.experiment_012.analysis import fit_scaling


def test_scaling_fit_distinguishes_linear_from_constant_and_logarithmic() -> None:
    result = fit_scaling(
        [(2, 4.0), (8, 16.0), (32, 64.0), (128, 256.0), (512, 1024.0)],
        branch_factor=8,
    )

    assert result["best_relationship"] == "linear_n"
    linear = next(item for item in result["fits"] if item["relationship"] == "linear_n")
    constant = next(item for item in result["fits"] if item["relationship"] == "constant")
    logarithmic = next(item for item in result["fits"] if item["relationship"] == "log2_n")
    assert linear["coefficient"] == 2.0
    assert linear["rss"] == 0.0
    assert linear["aicc"] < logarithmic["aicc"]
    assert linear["aicc"] < constant["aicc"]


def test_scaling_fit_classifies_a_bounded_metric_as_constant() -> None:
    result = fit_scaling(
        [(8, 8.0), (32, 8.0), (128, 8.0), (512, 8.0), (1000, 8.0)],
        branch_factor=8,
    )

    assert result["best_relationship"] == "constant"


def test_three_scale_fit_uses_cross_validation_when_aicc_is_not_comparable() -> None:
    result = fit_scaling([(8, 16.0), (32, 64.0), (128, 256.0)], branch_factor=8)

    assert result["ranking_criterion"] == "leave_one_out_rmse"
    assert result["best_relationship"] == "linear_n"
