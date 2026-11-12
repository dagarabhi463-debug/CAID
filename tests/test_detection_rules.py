
"""
CAID - Canvas AI Intrusion Detection System
Milestone 3: Safe Rule-Based Detection Engine

Features:
- Default detection rules
- Custom rule configuration
- Enable and disable rules
- Safe AST-based condition evaluation
- Single-event and DataFrame processing
- Rule statistics
- Compatibility with CAID Milestone 3 tests
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import ast
import math
import operator

import pandas as pd


# ============================================================
# ENUMS
# ============================================================

class RuleSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleAction(Enum):
    ALERT = "ALERT"
    LOG = "LOG"
    BLOCK = "BLOCK"
    INVESTIGATE = "INVESTIGATE"


# ============================================================
# EXCEPTIONS
# ============================================================

class RuleConfigurationError(ValueError):
    """Raised when a rule contains invalid configuration."""


# ============================================================
# DETECTION RULE MODEL
# ============================================================

@dataclass
class DetectionRule:
    rule_id: str
    name: str
    description: str
    condition: str
    severity: RuleSeverity
    action: RuleAction = RuleAction.ALERT
    threshold: float = 1.0
    enabled: bool = True
    weight: float = 0.1


# ============================================================
# SAFE CONDITION EVALUATOR
# ============================================================

class SafeConditionEvaluator:
    """
    Evaluates a restricted subset of Python expressions.

    Supported:
    - Comparisons
    - and / or / not
    - Lists and tuples
    - Numeric and string constants
    - Event fields
    - The special variable 'threshold'

    Does not allow function calls, imports, attribute access,
    indexing, eval(), or exec().
    """

    ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Name,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Load,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    )

    OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }

    def __init__(self, condition: str):
        if not isinstance(condition, str) or not condition.strip():
            raise RuleConfigurationError(
                "Condition must be a non-empty string."
            )

        try:
            self.tree = ast.parse(condition, mode="eval")
        except SyntaxError as exc:
            raise RuleConfigurationError(
                f"Invalid condition: {exc}"
            ) from exc

        self._validate(self.tree)

    def _validate(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED_NODES):
                raise RuleConfigurationError(
                    f"Unsafe or unsupported expression: "
                    f"{type(node).__name__}"
                )

            if isinstance(node, ast.Name):
                if node.id.startswith("__"):
                    raise RuleConfigurationError(
                        "Private names are not allowed."
                    )

            if isinstance(node, ast.Constant):
                if not isinstance(
                    node.value,
                    (str, int, float, bool, type(None))
                ):
                    raise RuleConfigurationError(
                        "Unsupported constant."
                    )

                if isinstance(node.value, float):
                    if not math.isfinite(node.value):
                        raise RuleConfigurationError(
                            "Non-finite constants are not allowed."
                        )

    def evaluate(
        self,
        row: Dict[str, Any],
        threshold: float = 1.0
    ) -> bool:
        try:
            result = self._evaluate_node(
                self.tree.body,
                row,
                threshold
            )
            return bool(result)

        except (
            TypeError,
            ValueError,
            KeyError,
            ZeroDivisionError
        ):
            return False

    def _evaluate_node(
        self,
        node: ast.AST,
        row: Dict[str, Any],
        threshold: float
    ) -> Any:

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id == "threshold":
                return threshold
            return row.get(node.id)

        if isinstance(node, ast.List):
            return [
                self._evaluate_node(item, row, threshold)
                for item in node.elts
            ]

        if isinstance(node, ast.Tuple):
            return tuple(
                self._evaluate_node(item, row, threshold)
                for item in node.elts
            )

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(
                    bool(self._evaluate_node(item, row, threshold))
                    for item in node.values
                )

            if isinstance(node.op, ast.Or):
                return any(
                    bool(self._evaluate_node(item, row, threshold))
                    for item in node.values
                )

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not bool(
                    self._evaluate_node(
                        node.operand, row, threshold
                    )
                )

        if isinstance(node, ast.Compare):
            left = self._evaluate_node(
                node.left, row, threshold
            )

            for op_node, comparator in zip(
                node.ops, node.comparators
            ):
                right = self._evaluate_node(
                    comparator, row, threshold
                )

                if left is None or right is None:
                    return False

                operation = self.OPERATORS.get(type(op_node))

                if operation is None:
                    return False

                if not operation(left, right):
                    return False

                left = right

            return True

        raise RuleConfigurationError(
            f"Unsupported expression: {type(node).__name__}"
        )


# ============================================================
# DETECTION RULES ENGINE
# ============================================================

class DetectionRulesEngine:

    def __init__(
        self,
        rules: Optional[List[DetectionRule]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ):
        self.rules: Dict[str, DetectionRule] = {}
        self.compiled_conditions: Dict[str, SafeConditionEvaluator] = {}
        self.rule_statistics: Dict[str, int] = {}

        # Load default rules unless a complete rule list was supplied.
        if rules is None:
            for rule in self._create_default_rules():
                self.add_rule(rule)
        else:
            for rule in rules:
                self.add_rule(rule)

        # Custom rules are added alongside default rules.
        if custom_rules:
            for custom_rule in custom_rules:
                self.add_rule_from_dict(custom_rule)

    # --------------------------------------------------------
    # DEFAULT RULES
    # --------------------------------------------------------

    def _create_default_rules(self) -> List[DetectionRule]:

        return [
            DetectionRule(
                rule_id="api_high_volume",
                name="High API Request Volume",
                description="Detects unusually high API request activity.",
                condition="api_requests > threshold",
                severity=RuleSeverity.HIGH,
                action=RuleAction.ALERT,
                threshold=1000,
                weight=0.20
            ),

            DetectionRule(
                rule_id="data_volume_high",
                name="High Data Transfer",
                description="Detects unusually large data transfers.",
                condition="data_mb > threshold",
                severity=RuleSeverity.HIGH,
                action=RuleAction.ALERT,
                threshold=500,
                weight=0.20
            ),

            DetectionRule(
                rule_id="courses_threshold",
                name="Unusual Course Access",
                description="Detects access to an unusually large number of courses.",
                condition="courses_accessed > threshold",
                severity=RuleSeverity.MEDIUM,
                threshold=10,
                weight=0.10
            ),

            DetectionRule(
                rule_id="accounts_threshold",
                name="Multiple Account Access",
                description="Detects unusually high account access.",
                condition="accounts_accessed > threshold",
                severity=RuleSeverity.HIGH,
                threshold=5,
                weight=0.15
            ),

            DetectionRule(
                rule_id="after_hours_activity",
                name="After-Hours Activity",
                description="Detects activity outside normal hours.",
                condition="after_hours == 1",
                severity=RuleSeverity.MEDIUM,
                threshold=1,
                weight=0.10
            ),

            DetectionRule(
                rule_id="suspicious_ip_class",
                name="Suspicious IP Classification",
                description="Detects events with a suspicious demo IP label.",
                condition=(
                    "ip_class == 'suspicious' or "
                    "ip_class == 'high_risk'"
                ),
                severity=RuleSeverity.HIGH,
                weight=0.20
            ),

            DetectionRule(
                rule_id="bulk_download_pattern",
                name="Bulk Download Pattern",
                description="Detects bulk download event types.",
                condition=(
                    "event_type in "
                    "['bulk_download', 'mass_download']"
                ),
                severity=RuleSeverity.HIGH,
                weight=0.20
            ),

            DetectionRule(
                rule_id="old_token_activity",
                name="Old Token Activity",
                description="Detects activity involving an old token.",
                condition="token_age_minutes > threshold",
                severity=RuleSeverity.MEDIUM,
                threshold=1440,
                weight=0.05
            )
        ]

    # --------------------------------------------------------
    # RULE CREATION
    # --------------------------------------------------------

    def add_rule(self, rule: DetectionRule) -> None:

        if not isinstance(rule, DetectionRule):
            raise RuleConfigurationError(
                "Expected a DetectionRule object."
            )

        if not rule.rule_id or not rule.rule_id.strip():
            raise RuleConfigurationError(
                "Rule ID cannot be empty."
            )

        if rule.rule_id in self.rules:
            raise RuleConfigurationError(
                f"Duplicate rule ID: {rule.rule_id}"
            )

        if not isinstance(rule.severity, RuleSeverity):
            raise RuleConfigurationError(
                "Invalid rule severity."
            )

        if not isinstance(rule.action, RuleAction):
            raise RuleConfigurationError(
                "Invalid rule action."
            )

        try:
            weight = float(rule.weight)
            threshold = float(rule.threshold)
        except (TypeError, ValueError) as exc:
            raise RuleConfigurationError(
                "Threshold and weight must be numeric."
            ) from exc

        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise RuleConfigurationError(
                "Weight must be between 0 and 1."
            )

        if not math.isfinite(threshold):
            raise RuleConfigurationError(
                "Threshold must be finite."
            )

        compiled = SafeConditionEvaluator(rule.condition)

        self.rules[rule.rule_id] = rule
        self.compiled_conditions[rule.rule_id] = compiled
        self.rule_statistics[rule.rule_id] = 0

    def add_rule_from_dict(self, data: Dict[str, Any]) -> None:

        try:
            severity = data.get(
                "severity", RuleSeverity.MEDIUM
            )
            action = data.get(
                "action", RuleAction.ALERT
            )

            if isinstance(severity, str):
                severity = RuleSeverity(severity.upper())

            if isinstance(action, str):
                action = RuleAction(action.upper())

            rule = DetectionRule(
                rule_id=data["rule_id"],
                name=data["name"],
                description=data.get("description", ""),
                condition=data["condition"],
                severity=severity,
                action=action,
                threshold=data.get("threshold", 1.0),
                enabled=data.get("enabled", True),
                weight=data.get("weight", 0.1)
            )

            self.add_rule(rule)

        except KeyError as exc:
            raise RuleConfigurationError(
                f"Missing required custom rule field: {exc}"
            ) from exc

    # --------------------------------------------------------
    # RULE CONFIGURATION
    # --------------------------------------------------------

    def get_rule(self, rule_id: str) -> Optional[DetectionRule]:
        return self.rules.get(rule_id)

    def configure_rule(
        self,
        rule_id: str,
        **changes: Any
    ) -> bool:

        rule = self.rules.get(rule_id)

        if rule is None:
            return False

        allowed = {
            "name",
            "description",
            "condition",
            "severity",
            "action",
            "threshold",
            "enabled",
            "weight"
        }

        if not set(changes).issubset(allowed):
            raise RuleConfigurationError(
                "Unsupported rule configuration field."
            )

        updated = DetectionRule(
            rule_id=rule.rule_id,
            name=changes.get("name", rule.name),
            description=changes.get(
                "description", rule.description
            ),
            condition=changes.get(
                "condition", rule.condition
            ),
            severity=changes.get(
                "severity", rule.severity
            ),
            action=changes.get("action", rule.action),
            threshold=changes.get(
                "threshold", rule.threshold
            ),
            enabled=changes.get("enabled", rule.enabled),
            weight=changes.get("weight", rule.weight)
        )

        if isinstance(updated.severity, str):
            updated.severity = RuleSeverity(
                updated.severity.upper()
            )

        if isinstance(updated.action, str):
            updated.action = RuleAction(
                updated.action.upper()
            )

        # Validate before replacing the existing rule.
        compiled = SafeConditionEvaluator(updated.condition)

        try:
            weight = float(updated.weight)
            threshold = float(updated.threshold)
        except (TypeError, ValueError) as exc:
            raise RuleConfigurationError(
                "Threshold and weight must be numeric."
            ) from exc

        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise RuleConfigurationError(
                "Weight must be between 0 and 1."
            )

        if not math.isfinite(threshold):
            raise RuleConfigurationError(
                "Threshold must be finite."
            )

        if not isinstance(updated.severity, RuleSeverity):
            raise RuleConfigurationError("Invalid severity.")

        if not isinstance(updated.action, RuleAction):
            raise RuleConfigurationError("Invalid action.")

        self.rules[rule_id] = updated
        self.compiled_conditions[rule_id] = compiled

        return True

    def enable_rule(self, rule_id: str) -> bool:

        if rule_id not in self.rules:
            return False

        self.rules[rule_id].enabled = True
        return True

    def disable_rule(self, rule_id: str) -> bool:

        if rule_id not in self.rules:
            return False

        self.rules[rule_id].enabled = False
        return True

    # --------------------------------------------------------
    # SINGLE EVENT EVALUATION
    # --------------------------------------------------------

    def check_rules(self, event: Any):
        """
        Evaluate rules against one event.

        Returns:
            score: float
            triggered: list of triggered rule dictionaries

        This tuple format matches the Milestone 3 tests.
        """

        if isinstance(event, pd.Series):
            row = event.to_dict()
        elif isinstance(event, dict):
            row = event
        else:
            raise TypeError(
                "Event must be a dictionary or pandas Series."
            )

        score = 0.0
        triggered = []

        for rule_id, rule in self.rules.items():

            if not rule.enabled:
                continue

            evaluator = self.compiled_conditions[rule_id]

            matched = evaluator.evaluate(
                row,
                threshold=float(rule.threshold)
            )

            if matched:
                triggered.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity.value,
                    "action": rule.action.value,
                    "weight": float(rule.weight)
                })

                score += float(rule.weight)
                self.rule_statistics[rule_id] += 1

        score = min(max(score, 0.0), 1.0)

        return score, triggered

    # --------------------------------------------------------
    # DATAFRAME PROCESSING
    # --------------------------------------------------------

    def process_events(
        self,
        events: pd.DataFrame
    ) -> pd.DataFrame:

        if not isinstance(events, pd.DataFrame):
            raise TypeError(
                "events must be a pandas DataFrame."
            )

        result = events.copy()

        if result.empty:
            result["rule_detection_score"] = pd.Series(
                dtype=float
            )
            result["triggered_rules"] = pd.Series(
                dtype=object
            )
            result["rule_score"] = pd.Series(dtype=float)
            result["rules_triggered"] = pd.Series(dtype=int)
            result["is_suspicious"] = pd.Series(dtype=bool)
            return result

        scores = []
        all_triggered = []

        for _, row in result.iterrows():
            score, triggered = self.check_rules(row)

            scores.append(score)
            all_triggered.append(triggered)

        result["rule_detection_score"] = scores

        # Keep rule_score as a compatibility alias.
        result["rule_score"] = scores

        result["triggered_rules"] = all_triggered

        result["rules_triggered"] = [
            len(items) for items in all_triggered
        ]

        result["is_suspicious"] = [
            len(items) > 0 for items in all_triggered
        ]

        return result

    # --------------------------------------------------------
    # RULE STATISTICS
    # --------------------------------------------------------

    def get_rule_statistics(
        self,
        processed_events: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Return rule trigger statistics.

        When a processed DataFrame is supplied, calculate
        event-level statistics from that DataFrame.
        """

        statistics = {
            "total_events": 0,
            "events_with_triggers": 0,
            "total_rule_triggers": 0,
            "rules": dict(self.rule_statistics)
        }

        if processed_events is None:
            return statistics

        if not isinstance(processed_events, pd.DataFrame):
            raise TypeError(
                "processed_events must be a pandas DataFrame."
            )

        statistics["total_events"] = len(processed_events)

        if "triggered_rules" in processed_events.columns:

            triggered_column = processed_events[
                "triggered_rules"
            ]

            counts = triggered_column.apply(
                lambda value: len(value)
                if isinstance(value, (list, tuple))
                else 0
            )

            statistics["events_with_triggers"] = int(
                (counts > 0).sum()
            )

            statistics["total_rule_triggers"] = int(
                counts.sum()
            )

        elif "rule_detection_score" in processed_events.columns:

            scores = pd.to_numeric(
                processed_events["rule_detection_score"],
                errors="coerce"
            ).fillna(0)

            statistics["events_with_triggers"] = int(
                (scores > 0).sum()
            )

        return statistics

    def get_enabled_rules(self) -> List[DetectionRule]:

        return [
            rule
            for rule in self.rules.values()
            if rule.enabled
        ]
