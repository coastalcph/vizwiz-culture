import json
import re
from typing import Any, Dict, Optional

from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str
from pydantic import BaseModel


def parse_json(text_response: str) -> Optional[Dict[str, Any]]:
    pattern = r"\{[^{}]*\}"
    matches = list(re.finditer(pattern, text_response))

    if not matches:
        return None

    match = matches[0]
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Extend the search for nested structures
        extended_json_str = _extend_search(text_response, match.span())
        try:
            return json.loads(extended_json_str)
        except json.JSONDecodeError:
            # If all else fails, try to extract the JSON data manually
            data = {}
            pairs = re.findall(r'(".*?")\s*:\s*(.*)', text_response)

            if not pairs:
                return None

            for pair in pairs:
                key = pair[0].strip('"')
                value = pair[1].strip('"')
                data[key] = value
            return data


def _extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            nest_count += 1
        elif text[i] == "}":
            nest_count -= 1
            if nest_count == 0:
                return text[start : i + 1]
    return text[start:end]


def validate_json_with_schema(json_data: Dict[str, Any], schema: BaseModel) -> bool:
    regex = build_regex_from_schema(convert_json_schema_to_str(schema), whitespace_pattern=r" ?")
    return re.fullmatch(regex, json.dumps(json_data)) is not None


def parse_pydantic_schema(pydantic_model: BaseModel):
    simple_schema = {}
    raw_schema = pydantic_model.model_json_schema()

    for name, value in raw_schema["properties"].items():
        # For boolean types, we want to display "true/false" instead of a description
        if "type" in value and value["type"] == "boolean":
            simple_schema[name] = "true/false"
        elif "description" in value:
            simple_schema[name] = value["description"]
        else:
            simple_schema[name] = f"<{name}>"

    return json.dumps(simple_schema, indent=2).replace('"true/false"', "true/false")
