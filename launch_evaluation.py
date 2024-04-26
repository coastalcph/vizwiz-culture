import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import pandas as pd
from eval.coco import COCO
from eval.eval import COCOEvalCap
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm


def load_preds_from_csv(predictions_csv_file: str) -> List[Dict[str, str]]:
    res = []
    df = pd.read_csv(predictions_csv_file, sep="\t")
    for image_path, response in zip(df["image_path"], df["response"]):
        res.append({"image_id": Path(image_path).stem, "caption": response})

    return res


def load_annotations(captions_file: str) -> Tuple[pd.DataFrame, List[int]]:
    data = []
    groups = []
    group2id = {}

    with open(captions_file) as f:
        for line in f:
            j = json.loads(line)

            group = j["image/locale"]
            if group not in group2id:
                group2id[group] = len(groups)
                groups.append(group)
            group_id = group2id[group]

            data.append((j["image/key"], group, group_id, j["en"]["caption"][0]))
            data.append((j["image/key"], group, group_id, j["en"]["caption"][1]))

    df = pd.DataFrame(data, columns=["id", "culture", "group", "caption"])

    return df, groups


def run_coco_eval(anns: Dict[str, Any], preds: List[Dict[str, str]]) -> Dict[str, float]:
    coco = COCO(anns)
    coco_res = coco.loadRes(preds)

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()

    # evaluate results
    coco_eval.evaluate()

    return coco_eval.eval


def get_coco_anns_json_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    anns_data = {"type": "captions", "images": [], "annotations": [], "info": {}, "licenses": []}

    for id, cap in zip(df["id"], df["caption"]):
        anns_data["images"].append({"id": id})
        anns_data["annotations"].append(
            {"image_id": id, "id": len(anns_data["annotations"]), "caption": cap}
        )

    return anns_data


def save_results_json(results: Dict[str, Any], file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, sort_keys=True)


def run_evaluation(
    annot_df: pd.DataFrame, preds_list: List[Dict[str, str]], output_dir: Path, groups: List[str]
) -> None:
    """
    Args:
        annot_df: DataFrame with the following columns: id, culture, group, caption
        preds_list: List of dictionaries with the following keys: image_id, caption
        output_dir: Path to the output directory
        groups: List of group names

    Returns:
        None
    """

    output_dir.mkdir(exist_ok=True)

    # Create a mapping from image id to group name
    group_mapping = {id: group for id, group in zip(annot_df["id"], annot_df["group"])}

    # Split predictions into groups
    groups_preds_data = defaultdict(list)
    for entry in preds_list:
        groups_preds_data[group_mapping[entry["image_id"]]].append(entry)

    # Compute scores for each group
    pbar = tqdm(range(len(groups)))
    for gid in pbar:
        pbar.set_description(f"Computing scores: [{groups[gid]}]")

        group_df = annot_df.loc[annot_df.group == gid]

        group_anns_data = get_coco_anns_json_from_df(group_df)
        group_preds_data = groups_preds_data[gid]

        # Run evaluation
        group_results = run_coco_eval(group_anns_data, group_preds_data)
        group_results.update(
            {
                "group_name": groups[gid],
                "num_predictions": len(group_preds_data),
                "num_annotations": len(group_anns_data["annotations"]),
            }
        )

        save_results_json(group_results, output_dir / f"scores_group_{gid}.json")

    print("Computing full scores")

    full_anns = get_coco_anns_json_from_df(annot_df)
    full_preds = preds_list

    full_results = run_coco_eval(full_anns, full_preds)
    full_results.update(
        {
            "group_name": "all",
            "num_predictions": len(full_preds),
            "num_annotations": len(full_anns["annotations"]),
        }
    )
    save_results_json(full_results, output_dir / f"scores_full.json")

    print(
        f"\nFull results:\n{json.dumps(full_results, indent=4, ensure_ascii=False, sort_keys=True)}"
    )


@dataclass
class RunConfig:
    annotations_file: str
    predictions_csv: str
    output_dir: str


cs = ConfigStore.instance()
cs.store(name="base_config", node=RunConfig)


@hydra.main(version_base=None, config_name="base_config")
def main(config: DictConfig):
    # Load annotations into a dataframe
    annot_df, groups = load_annotations(config.annotations_file)

    # Load predictions
    predictions = load_preds_from_csv(config.predictions_csv)

    # Run evaluation
    run_evaluation(annot_df, predictions, Path(config.output_dir), groups)


if __name__ == "__main__":
    main()
