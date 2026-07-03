from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence


# ── Parameters ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NumericParameter:
    name: str
    min: float
    max: float
    step: Optional[float] = None
    default: Optional[float] = None


@dataclass(frozen=True)
class CategoricalParameter:
    name: str
    choices: List[str]
    default: Optional[str] = None


Parameter = NumericParameter | CategoricalParameter


# ── Search space ─────────────────────────────────────────────────────────────

@dataclass
class SearchSpace:
    parameters: List[Parameter]

    def numeric_parameters(self) -> list[NumericParameter]:
        return [p for p in self.parameters if isinstance(p, NumericParameter)]

    def categorical_parameters(self) -> list[CategoricalParameter]:
        return [p for p in self.parameters if isinstance(p, CategoricalParameter)]

    def parameter_by_name(self, name: str) -> Optional[Parameter]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None


# ── Objective ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    direction: Literal['maximize', 'minimize'] = 'maximize'
    weight: float = 1.0
    target_material: Optional[str] = None


class ObjectiveFunction:
    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def evaluate(self, result: dict) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f'{self.config.name}(direction={self.config.direction}'
            f'{f", target={self.config.target_material}" if self.config.target_material else ""})'
        )


# ── Constraints ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConstraintConfig:
    type: str
    params: Dict[str, Any]


class Constraint:
    def check(self, candidate: dict[str, float | str]) -> tuple[bool, list[str]]:
        raise NotImplementedError

    def satisfied(self, candidate: dict[str, float | str]) -> bool:
        ok, _ = self.check(candidate)
        return ok


# ── Solver ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SolverConfig:
    method: str
    n_evaluations: int = 50
    n_initial: int = 10
    seed: int = 0


@dataclass
class OptimizationResult:
    best_candidate: dict[str, float | str]
    best_value: float
    history: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0

    def summary(self) -> str:
        lines = [f'Best objective: {self.best_value:.4f}']
        lines.append(f'Best candidate: {self.best_candidate}')
        lines.append(f'Total evaluations: {len(self.history)}')
        lines.append(f'Wall time: {self.wall_time_s:.1f}s')
        return '\n'.join(lines)
