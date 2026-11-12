
"""
CAID - Canvas AI Intrusion Detection System
Safe rule-based threat detection engine.

Evaluates rule conditions using a restricted Python AST interpreter.
It does not use eval() or exec().
"""

from dataclasses import dataclass
from enum import Enum
import ast
import math
import operator
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


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


class RuleConfigurationError(ValueError):
    """Raised when a detection rule has an invalid or unsafe condition."""


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


def _compile_node(node: ast.AST):
    """Compile a validated AST node into ``fn(row, threshold) -> value``.

    The AST has already passed ``_SafeCondition._validate`` (allow-listed
    node types only — no calls, attribute access, imports, or indexing),
    so building plain Python closures over it cannot execute arbitrary
    code. This removes the per-node ``isinstance`` dispatch of the
    interpreter on the hot per-row path while preserving identical
    semantics, including short-circuiting, comparison chaining, and the
    ``None`` -> ``False`` comparison rule.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        return lambda row, thr: value

    if isinstance(node, ast.Name):
        if node.id == "threshold":
            return lambda row, thr: thr
        name = node.id
        return lambda row, thr: row.get(name)

    if isinstance(node, ast.List):
        fns = [_compile_node(e) for e in node.elts]
        return lambda row, thr: [f(row, thr) for f in fns]

    if isinstance(node, ast.Tuple):
        fns = [_compile_node(e) for e in node.elts]
        return lambda row, thr: tuple(f(row, thr) for f in fns)

    if isinstance(node, ast.Set):
        fns = [_compile_node(e) for e in node.elts]
        return lambda row, thr: {f(row, thr) for f in fns}

    if isinstance(node, ast.BoolOp):
        fns = [_compile_node(v) for v in node.values]
        if isinstance(node.op, ast.And):
            def _and(row, thr):
                for f in fns:
                    if not f(row, thr):
                        return False
                return True
            return _and
        if isinstance(node.op, ast.Or):
            def _or(row, thr):
                for f in fns:
                    if f(row, thr):
                        return True
                return False
            return _or
        raise RuleConfigurationError("Unsupported boolean operator.")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            operand = _compile_node(node.operand)
            return lambda row, thr: not bool(operand(row, thr))
        raise RuleConfigurationError("Unsupported unary operator.")

    if isinstance(node, ast.Compare):
        left_fn = _compile_node(node.left)
        ops = []
        for op_node, comparator in zip(node.ops, node.comparators):
            operation = _SafeCondition.COMPARISON_OPERATORS.get(
                type(op_node))
            if operation is None:
                raise RuleConfigurationError(
                    "Unsupported comparison operator.")
            ops.append((operation, _compile_node(comparator)))

        def _compare(row, thr):
            left = left_fn(row, thr)
            for operation, right_fn in ops:
                right = right_fn(row, thr)
                if left is None or right is None:
                    return False
                if not operation(left, right):
                    return False
                left = right
            return True
        return _compare

    raise RuleConfigurationError(
        f"Unsupported expression node: {type(node).__name__}")


class _SafeCondition:
    """
    Safely evaluates a limited subset of Python expressions.

    Supported:
    - Comparisons: ==, !=, >, >=, <, <=
    - Membership: in, not in
    - Boolean expressions: and, or, not
    - Numeric, string and boolean constants
    - Lists, tuples and sets
    - Row field names
    - The special variable 'threshold'

    Function calls, imports, attribute access, indexing and
    arbitrary Python execution are not permitted.
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
        ast.Set,
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

    COMPARISON_OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.In: lambda left, right: left in right,
        ast.NotIn: lambda left, right: left not in right,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }

    def __init__(self, condition: str):
        if not isinstance(condition, str) or not condition.strip():
            raise RuleConfigurationError(
                "Rule condition must be a non-empty string."
            )

        try:
            self.tree = ast.parse(condition, mode="eval")
        except SyntaxError as exc:
            raise RuleConfigurationError(
                f"Invalid rule condition: {exc}"
            ) from exc

        self._validate(self.tree)
        self._fn = _compile_node(self.tree.body)

    def _validate(self, node: ast.AST) -> None:
        """Reject every AST node outside the explicit allow-list."""

        for child in ast.walk(node):
            if not isinstance(child, self.ALLOWED_NODES):
                raise RuleConfigurationError(
                    f"Unsupported expression element: "
                    f"{type(child).__name__}"
                )

            if isinstance(child, ast.Name):
                if child.id.startswith("__"):
                    raise RuleConfigurationError(
                        "Private or special names are not allowed."
                    )

            if isinstance(child, ast.Constant):
                if not isinstance(
                    child.value,
                    (str, int, float, bool, type(None)),
                ):
                    raise RuleConfigurationError(
                        "Unsupported constant type."
                    )

                if isinstance(child.value, float):
                    if not math.isfinite(child.value):
                        raise RuleConfigurationError(
                            "Non-finite numbers are not allowed."
                        )

    def evaluate(
        self,
        row: Dict[str, Any],
        threshold: float = 1.0,
    ) -> bool:
        """Evaluate the condition against one event row."""

        try:
            result = self._fn(row, threshold)
            return bool(result)
        except (
            TypeError,
            ValueError,
            KeyError,
            ZeroDivisionError,
        ):
            # Invalid or missing event data should not trigger a rule.
            return False

    def _evaluate_node(
        self,
        node: ast.AST,
        row: Dict[str, Any],
        threshold: float,
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

        if isinstance(node, ast.Set):
            return {
                self._evaluate_node(item, row, threshold)
                for item in node.elts
            }

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not bool(
                        self._evaluate_node(value, row, threshold)
                    ):
                        return False
                return True

            if isinstance(node.op, ast.Or):
                for value in node.values:
                    if bool(
                        self._evaluate_node(value, row, threshold)
                    ):
                        return True
                return False

            raise RuleConfigurationError(
                "Unsupported boolean operator."
            )

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not bool(
                    self._evaluate_node(node.operand, row, threshold)
                )

            raise RuleConfigurationError(
                "Unsupported unary operator."
            )

        if isinstance(node, ast.Compare):
            left = self._evaluate_node(
                node.left,
                row,
                threshold,
            )

            for op_node, comparator in zip(
                node.ops,
                node.comparators,
            ):
                right = self._evaluate_node(
                    comparator,
                    row,
                    threshold,
                )

                operation = self.COMPARISON_OPERATORS.get(
                    type(op_node)
                )

                if operation is None:
                    raise RuleConfigurationError(
                        "Unsupported comparison operator."
                    )

                if left is None or right is None:
                    return False

                if not operation(left, right):
                    return False

                left = right

            return True

        raise RuleConfigurationError(
            f"Unsupported expression node: {type(node).__name__}"
        )


class DetectionRulesEngine:
    """Manages and evaluates CAID detection rules."""

    def __init__(self, rules: Optional[List[DetectionRule]] = None):
        self.rules: Dict[str, DetectionRule] = {}
        self._compiled_conditions: Dict[str, _SafeCondition] = {}
        self.rule_statistics: Dict[str, int] = {}

        if rules is None:
            rules = self._create_default_rules()

        for rule in rules:
            self.add_rule(rule)

    def _create_default_rules(self) -> List[DetectionRule]:
        """Create the default CAID security detection rules."""

        return [
            DetectionRule(
                rule_id="high_api_requests",
                name="High API Request Volume",
                description="Detects unusually high API request activity.",
                condition="api_requests > threshold",
                severity=RuleSeverity.MEDIUM,
                threshold=100,
                weight=0.15,
            ),
            DetectionRule(
                rule_id="after_hours_activity",
                name="After-Hours Activity",
                description="Detects activity outside normal hours.",
                condition="after_hours == 1",
                severity=RuleSeverity.MEDIUM,
                weight=0.10,
            ),
            DetectionRule(
                rule_id="bulk_download_pattern",
                name="Bulk Download Pattern",
                description="Detects bulk download event types.",
                condition="event_type in ['bulk_download', 'mass_download']",
                severity=RuleSeverity.HIGH,
                weight=0.20,
            ),
            DetectionRule(
                rule_id="high_data_transfer",
                name="High Data Transfer",
                description="Detects unusually large data transfers.",
                condition="data_mb > threshold",
                severity=RuleSeverity.HIGH,
                threshold=500,
                weight=0.15,
            ),
            DetectionRule(
                rule_id="multiple_accounts_accessed",
                name="Multiple Account Access",
                description="Detects access to many accounts in one event.",
                condition="accounts_accessed > threshold",
                severity=RuleSeverity.HIGH,
                threshold=10,
                weight=0.10,
            ),
            DetectionRule(
                rule_id="old_token_activity",
                name="Old Token Activity",
                description="Detects activity involving an old token.",
                condition="token_age_minutes > threshold",
                severity=RuleSeverity.MEDIUM,
                threshold=1440,
                weight=0.05,
            ),
            DetectionRule(
                rule_id="critical_access_pattern",
                name="Critical Access Pattern",
                description="Detects unusually high account access.",
                condition=(
                    "accounts_accessed > 25 and "
                    "data_mb > 100"
                ),
                severity=RuleSeverity.CRITICAL,
                weight=0.20,
            ),
        ]

    def add_rule(self, rule: DetectionRule) -> None:
        """Validate and add a detection rule."""

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
                "Rule weight and threshold must be numeric."
            ) from exc

        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise RuleConfigurationError(
                "Rule weight must be between 0 and 1."
            )

        if not math.isfinite(threshold):
            raise RuleConfigurationError(
                "Rule threshold must be finite."
            )

        compiled = _SafeCondition(rule.condition)

        self.rules[rule.rule_id] = rule
        self._compiled_conditions[rule.rule_id] = compiled
        self.rule_statistics[rule.rule_id] = 0

    def get_rule(self, rule_id: str) -> Optional[DetectionRule]:
        return self.rules.get(rule_id)

    def configure_rule(
        self,
        rule_id: str,
        **changes: Any,
    ) -> bool:
        """Update a rule and revalidate its condition."""

        rule = self.rules.get(rule_id)

        if rule is None:
            return False

        allowed_fields = {
            "name",
            "description",
            "condition",
            "severity",
            "action",
            "threshold",
            "enabled",
            "weight",
        }

        if not set(changes).issubset(allowed_fields):
            raise RuleConfigurationError(
                "One or more unsupported rule fields were supplied."
            )

        updated = DetectionRule(
            rule_id=rule.rule_id,
            name=changes.get("name", rule.name),
            description=changes.get("description", rule.description),
            condition=changes.get("condition", rule.condition),
            severity=changes.get("severity", rule.severity),
            action=changes.get("action", rule.action),
            threshold=changes.get("threshold", rule.threshold),
            enabled=changes.get("enabled", rule.enabled),
            weight=changes.get("weight", rule.weight),
        )

        compiled = _SafeCondition(updated.condition)

        weight = float(updated.weight)
        threshold = float(updated.threshold)

        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise RuleConfigurationError(
                "Rule weight must be between 0 and 1."
            )

        if not math.isfinite(threshold):
            raise RuleConfigurationError(
                "Rule threshold must be finite."
            )

        if not isinstance(updated.severity, RuleSeverity):
            raise RuleConfigurationError("Invalid rule severity.")

        if not isinstance(updated.action, RuleAction):
            raise RuleConfigurationError("Invalid rule action.")

        self.rules[rule_id] = updated
        self._compiled_conditions[rule_id] = compiled
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

    def _create_condition_lambda(
        self,
        condition: str,
        threshold: float = 1.0,
    ) -> Callable[[Dict[str, Any]], bool]:
        """
        Compatibility helper for callers using the earlier method.

        Returns a safe callable; does not execute arbitrary Python.
        """

        compiled = _SafeCondition(condition)

        return lambda row: compiled.evaluate(
            row,
            threshold=threshold,
        )

    def check_rules(
        self,
        event: Union[Dict[str, Any], pd.Series],
    ) -> Dict[str, Any]:
        """Evaluate all enabled rules against a single event."""

        if isinstance(event, pd.Series):
            row = event.to_dict()
        elif isinstance(event, dict):
            row = event
        else:
            raise TypeError(
                "Event must be a dictionary or pandas Series."
            )

        triggered_rules = []
        total_score = 0.0

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            condition = self._compiled_conditions[rule_id]

            if condition.evaluate(
                row,
                threshold=float(rule.threshold),
            ):
                triggered_rules.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity.value,
                    "action": rule.action.value,
                    "weight": float(rule.weight),
                })

                self.rule_statistics[rule_id] += 1
                total_score += float(rule.weight)

        # Keep the combined rule score within the range 0 to 1.
        total_score = min(max(total_score, 0.0), 1.0)

        return {
            "triggered_rules": triggered_rules,
            "rule_score": total_score,
            "rules_triggered": len(triggered_rules),
            "is_suspicious": len(triggered_rules) > 0,
        }

    def process_events(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluate rules for every row in a DataFrame."""

        if not isinstance(events, pd.DataFrame):
            raise TypeError("events must be a pandas DataFrame.")

        result = events.copy()

        if result.empty:
            result["triggered_rules"] = pd.Series(dtype=object)
            result["rule_score"] = pd.Series(dtype=float)
            result["rules_triggered"] = pd.Series(dtype=int)
            result["is_suspicious"] = pd.Series(dtype=bool)
            return result

        # to_dict('records') is far cheaper than iterrows(): plain dicts
        # feed check_rules directly (it accepts dict or Series).
        evaluations = [
            self.check_rules(row)
            for row in result.to_dict('records')
        ]

        result["triggered_rules"] = [
            item["triggered_rules"]
            for item in evaluations
        ]

        result["rule_score"] = [
            item["rule_score"]
            for item in evaluations
        ]

        result["rules_triggered"] = [
            item["rules_triggered"]
            for item in evaluations
        ]

        result["is_suspicious"] = [
            item["is_suspicious"]
            for item in evaluations
        ]

        return result

    def get_enabled_rules(self) -> List[DetectionRule]:
        return [
            rule
            for rule in self.rules.values()
            if rule.enabled
        ]

    def get_rule_statistics(self) -> Dict[str, int]:
        return dict(self.rule_statistics)
