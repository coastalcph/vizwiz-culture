import csv
import json
import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import wandb

from ..dataset.dataset_base import Dataset
from ..modeling.modeling_base import VisionLanguageModel
from .completion import Completion
from .misc import get_random_name
from .usage_tracking import CostSummary, UsageTracker

logger = logging.getLogger(__name__)


class Callback(ABC):
    def __init__(self, model: VisionLanguageModel, dataset: Dataset, usage_tracker: UsageTracker):
        self.model = model
        self.dataset = dataset
        self.usage_tracker = usage_tracker

    @abstractmethod
    def on_step_end(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def on_run_end(self) -> None: ...


class SaveToCsvCallback(Callback):
    def __init__(self, file_path: Union[str, Path], **kwargs):
        super().__init__(**kwargs)

        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.file_path.exists() and self.file_path.is_file():
            raise FileExistsError(
                f"CSV file '{str(self.file_path)}' already exists, will not overwrite."
            )

    def on_step_end(self, step_index: int, completion: Completion):
        with open(self.file_path, "a") as f:
            writer = csv.writer(f, delimiter="\t", quoting=3, escapechar="\\", quotechar='"')
            if step_index == 0:
                writer.writerow(completion.keys_as_list())
            writer.writerow(completion.values_as_list())

    def on_run_end(self) -> None: ...


class LoggingCallback(Callback):
    def on_step_end(self, step_index: int, completion: Completion):
        completion_dict = {
            k: v for k, v in zip(completion.keys_as_list(), completion.values_as_list())
        }
        logger.info(
            f"\n{'-'*50}\nStep {step_index}:\n{json.dumps(completion_dict, indent=4, ensure_ascii=False)}\n{'-'*50}\n"
        )

    def on_run_end(self) -> None: ...


class WandbCallback(Callback):
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        table_name: str = "results",
        log_every: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.project = project
        self.entity = entity
        self.run_name = run_name or get_random_name()
        self.table_name = table_name
        self.log_every = log_every

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
        )

        self.table: Optional[wandb.Table] = None

    def on_step_end(self, step_index: int, completion: Completion):
        if self.table is None:
            self.table = wandb.Table(columns=completion.keys_as_list() + ["image"])
        else:
            self.table = wandb.Table(columns=self.table.columns, data=self.table.data)

        self.table.add_data(
            *completion.values_as_list(), wandb.Image(completion.example.image_path)
        )

        if (step_index + 1) % self.log_every == 0:
            wandb.log({f"Tables/{self.table_name}": self.table})

    def on_run_end(self):
        self.table = wandb.Table(columns=self.table.columns, data=self.table.data)
        wandb.log({f"Tables/{self.table_name}": self.table})

        self.run.finish()


class CostLoggingCallback(Callback):
    def __init__(self, log_every: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every

    def _calculate_costs(self) -> Optional[CostSummary]:
        if not hasattr(self.model, "pricing"):
            return None

        pricing = self.model.pricing

        input_cost_per_token = Decimal(pricing.usd_per_output_unit) / Decimal(pricing.unit_tokens)
        output_cost_per_token = Decimal(pricing.usd_per_output_unit) / Decimal(pricing.unit_tokens)

        input_cost = input_cost_per_token * Decimal(self.usage_tracker.input_token_count)
        output_cost = output_cost_per_token * Decimal(self.usage_tracker.output_token_count)
        return CostSummary(input_cost, output_cost)

    def on_step_end(self, step_index: int, completion: Completion):
        cost_summary = self._calculate_costs()

        if cost_summary is not None and (step_index + 1) % self.log_every == 0:
            logger.info(str(cost_summary))

            if wandb.run:
                wandb.log(cost_summary.asdict(), step=step_index)

    def on_run_end(self) -> None:
        cost_summary = self._calculate_costs()

        if cost_summary is not None:
            logger.info(str(cost_summary))

            if wandb.run:
                wandb.log(cost_summary.asdict())
