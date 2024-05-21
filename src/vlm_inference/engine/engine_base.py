import logging
import time
from functools import partial
from typing import Callable, List, Type

from pydantic import BaseModel as PydanticBaseModel
from tqdm import tqdm

from ..dataset.dataset_base import ImageDataset, ImageExample
from ..modeling.modeling_base import VisionLanguageModel
from ..utils.callbacks import Callback
from ..utils.completion import JsonCompletion, StringCompletion
from ..utils.json_parsing import parse_json, validate_json_with_schema
from ..utils.usage_tracking import UsageTracker

logger = logging.getLogger(__name__)


class Engine:
    """Engine class for all models."""

    def __init__(
        self, model: VisionLanguageModel, dataset: ImageDataset, callbacks: List[Callable] = []
    ):
        self.model = model
        self.dataset = dataset
        self.metadata_tracker = UsageTracker()

        callback_kwargs = dict(model=model, dataset=dataset, usage_tracker=self.metadata_tracker)
        self.callbacks: List[Callback] = [
            callback_cls(**callback_kwargs) for callback_cls in callbacks
        ]


    def json_step(
        self, example: ImageExample, json_schema: Type[PydanticBaseModel]
    ) -> JsonCompletion:
        for retry in range(self.model.gen_max_retries):
            try:
                generated_text, usage_metadata = self.model.generate(example, json_schema)
                generated_json = parse_json(generated_text)

                self.metadata_tracker.update(usage_metadata)
            except Exception as e:
                logger.warning(e)
                generated_json = None
                time.sleep(self.model.gen_sleep_duration)

            if generated_json is not None and validate_json_with_schema(
                generated_json, json_schema
            ):
                break

        if generated_json is None:
            logger.warning(
                f"Failed to generate JSON after {self.model.gen_max_retries} iterations"
            )
            return JsonCompletion(
                example=example,
                response={field: None for field in json_schema.model_fields.keys()},
            )

        logger.info(f"Generated JSON after {retry + 1} iterations")

        return JsonCompletion(example=example, response=generated_json)

    def step(self, example: ImageExample) -> StringCompletion:
        for retry in range(self.model.gen_max_retries):
            try:
                generated_text, usage_metadata = self.model.generate(example)

                self.metadata_tracker.update(usage_metadata)
            except Exception as e:
                logger.warning(e)
                generated_text = ""
                time.sleep(self.model.gen_sleep_duration)

            if generated_text.strip():
                break

        if not generated_text:
            logger.warning(
                f"Failed to generate text after {self.model.gen_max_retries} iterations"
            )
            return StringCompletion(example=example, response="")

        logger.info(f"Generated text after {retry + 1} iterations")

        return StringCompletion(example=example, response=generated_text)

    def run(self):
        step_cls = (
            partial(self.json_step, json_schema=self.dataset.json_schema)
            if self.model.json_mode
            else self.step
        )

        for i in tqdm(range(len(self.dataset))):
            completion = step_cls(self.dataset[i])
            for callback in self.callbacks:
                callback.on_step_end(i, completion)

        for callback in self.callbacks:
            callback.on_run_end()


def run_engine(model: VisionLanguageModel, dataset: ImageDataset, callbacks: List[Callback] = []):
    engine = Engine(model=model, dataset=dataset, callbacks=callbacks)
    engine.run()