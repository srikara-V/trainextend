"""Expand forecasting experiment configs into per-cutoff jobs."""

from __future__ import annotations

from trainextend.schemas import ExperimentConfig, TrainingConfig


def expand_experiment(config: ExperimentConfig) -> list[TrainingConfig]:
    """Turn one experiment YAML into one TrainingConfig per backtest cutoff."""
    jobs: list[TrainingConfig] = []
    rt = config.runtime
    for cutoff in config.backtest.cutoffs:
        jobs.append(
            TrainingConfig(
                task="forecast",
                name=f"{config.experiment_name}_{cutoff}",
                experiment_name=config.experiment_name,
                cutoff=cutoff,
                dataset_name=config.dataset.name,
                dataset_path=config.dataset.path,
                target=config.dataset.target,
                time_col=config.dataset.time_col,
                model_name=config.model.name,
                context_length=config.model.context_length,
                prediction_length=config.model.prediction_length,
                metric_names=list(config.metrics),
                gpu_type=rt.gpu,
                use_gpu=rt.use_gpu,
                max_steps=rt.max_steps,
                max_epochs=rt.max_epochs,
                batch_size=rt.batch_size,
                learning_rate=rt.learning_rate,
                checkpoint_every_steps=rt.checkpoint_every_steps,
                resume=rt.resume,
                simulate_failure_at_step=rt.simulate_failure_at_step,
            )
        )
    return jobs
