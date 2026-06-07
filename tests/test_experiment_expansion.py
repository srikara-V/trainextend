from __future__ import annotations

from trainextend.experiment import expand_experiment
from trainextend.schemas import BacktestSpec, DatasetSpec, ExperimentConfig, ModelSpec, RuntimeSpec


def test_expand_experiment_cutoffs():
    config = ExperimentConfig(
        experiment_name="demo",
        dataset=DatasetSpec(name="synthetic"),
        model=ModelSpec(context_length=96, prediction_length=24),
        backtest=BacktestSpec(cutoffs=["4000", "4200", "4400"]),
        runtime=RuntimeSpec(max_steps=100, use_gpu=False, gpu="cpu"),
    )
    jobs = expand_experiment(config)
    assert len(jobs) == 3
    assert all(j.task == "forecast" for j in jobs)
    assert {j.cutoff for j in jobs} == {"4000", "4200", "4400"}
    assert all(j.prediction_length == 24 for j in jobs)
    assert all(j.context_length == 96 for j in jobs)
