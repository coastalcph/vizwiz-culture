from dataclasses import dataclass
from decimal import Decimal


@dataclass
class UsageMetadata:
    input_token_count: int
    output_token_count: int


class UsageTracker:
    def __init__(self):
        self.input_token_count = 0
        self.output_token_count = 0

    def update(self, metadata: UsageMetadata):
        self.input_token_count += metadata.input_token_count
        self.output_token_count += metadata.output_token_count


@dataclass
class CostSummary:
    input_cost: Decimal
    output_cost: Decimal

    @property
    def total_cost(self):
        return self.input_cost + self.output_cost

    def __str__(self):
        max_width = len(f"{self.total_cost:.2f}")
        return f"""
Cost Summary:
+{'-' * (22 + max_width)}+
| {'Input Tokens': <14}: {self.input_cost:>{max_width}.2f} USD |
| {'Output Tokens': <14}: {self.output_cost:>{max_width}.2f} USD |
| {'-' * (20 + max_width)} |
| {'Total': <14}: {self.total_cost:>{max_width}.2f} USD |
+{'-' * (22 + max_width)}+
"""

    def asdict(self):
        return {
            "Cost/input_tokens": float(self.input_cost),
            "Cost/output_tokens": float(self.output_cost),
            "Cost/total": float(self.total_cost),
        }
