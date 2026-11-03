from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import wandb

type MetricReduction = Literal["mean"]


@dataclass
class BatchMetric:
    value: int
    num_samples: int


@dataclass
class MonitorMetric:
    name: str
    data: list[BatchMetric] = field(default_factory=list)

    def __reduce_mean(self) -> float:
        total_value = sum(batch.value for batch in self.data)
        total_samples = sum(batch.num_samples for batch in self.data)
        return total_value / total_samples if total_samples > 0 else 0.0

    def reduce(self, reduction: MetricReduction = "mean"):
        if reduction == "mean":
            return self.__reduce_mean()
        else:
            raise ValueError(f"Metric {reduction} is not supported")

    def append(self, value: int = None, num_samples: int = None) -> None:
        self.data.append(BatchMetric(value=value, num_samples=num_samples))


@dataclass
class MonitorConfig:
    project: str
    name: str
    log_step: int
    reduction: MetricReduction = "mean"
    entity: str | None = None


class Monitor:
    def __init__(
        self, config: MonitorConfig, run_config: Mapping[Any, Any] | None = None
    ):
        self.config = config

        wandb.init(
            project=config.project,
            name=config.name,
            entity=config.entity,
            config=run_config,
        )
        self.__zero()

    @property
    def global_samples(self) -> int:
        return self._global_samples

    @property
    def global_steps(self) -> int:
        return self._global_steps

    def __zero(self):
        self._global_samples: int = 0
        self._global_steps: int = 0
        self._metrics: Mapping[str, MonitorMetric] = {}

    def __reduce(self) -> Mapping[str, float]:
        data = {}
        while self._metrics:
            metric: MonitorMetric = self._metrics.popitem()[1]
            data[metric.name] = metric.reduce(reduction=self.config.reduction)
        assert self._metrics == {}, "Metrics should be empty after reduction"
        return data

    def log(self, data: Mapping[str, float], prefix: str | None = None):
        if prefix is not None:
            data = {f"{prefix}/{k}": v for k, v in data.items()}
        wandb.log(data, step=self.global_samples)

    def update(
        self,
        data: Mapping[str, float],
        num_samples: int | None = None,
        prefix: str | None = None,
    ):
        num_samples = num_samples or 0
        for k, v in data.items():
            if k not in self._metrics:
                self._metrics[k] = MonitorMetric(name=k)
            self._metrics[k].append(value=v, num_samples=num_samples)

        self._global_samples += num_samples
        self._global_steps += 1

        if (self.global_steps % self.config.log_step) == 0:
            data = self.__reduce()
            self.log(data, prefix=prefix)
