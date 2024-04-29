import csv
import json
import logging
from pathlib import Path
from typing import Optional, Union

import wandb

from ..engine.base import Completion
from .misc import get_random_name

logger = logging.getLogger(__name__)


class SaveToCsvCallback:
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.file_path.exists() and self.file_path.is_file():
            raise FileExistsError(
                f"CSV file '{str(self.file_path)}' already exists, will not overwrite."
            )

    def __call__(self, step_index: int, completion: Completion):
        with open(self.file_path, "a") as f:
            writer = csv.writer(f, delimiter="\t", quoting=3, escapechar="\\", quotechar='"')
            if step_index == 0:
                writer.writerow(completion.keys_as_list())
            writer.writerow(completion.values_as_list())


class LoggingCallback:
    def __call__(self, step_index: int, completion: Completion):
        completion_dict = {
            k: v for k, v in zip(completion.keys_as_list(), completion.values_as_list())
        }
        logger.info(
            f"\n{'-'*50}\nStep {step_index}:\n{json.dumps(completion_dict, indent=4, ensure_ascii=False)}\n{'-'*50}\n"
        )


class WandbCallback:
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        table_name: str = "results",
        log_every: int = 50,
    ):
        self.project = project
        self.entity = entity
        self.run_name = run_name or get_random_name()
        self.table_name = table_name
        self.log_every = log_every

        self.run = wandb.init(project=self.project, entity=self.entity, name=self.run_name)
        self.table = None

    def __call__(self, step_index: int, completion: Completion):
        if self.table is None:
            self.table = wandb.Table(columns=completion.keys_as_list() + ["image"])
        else:
            self.table = wandb.Table(columns=self.table.columns, data=self.table.data)

        self.table.add_data(
            *completion.values_as_list(), wandb.Image(completion.example.image_path)
        )

        if step_index % self.log_every == 0:
            wandb.log({f"Tables/{self.table_name}": self.table})
