from optimization.interfaces import ObjectiveFunction, ObjectiveConfig


class FullAbsorptionObjective(ObjectiveFunction):
    def evaluate(self, result: dict) -> float:
        if result.get('infeasible'):
            return -1.0 if self.config.direction == 'maximize' else 1e9
        return result.get('full_count', 0) / max(result.get('n_events', 1), 1)


class TotalDepositionObjective(ObjectiveFunction):
    def evaluate(self, result: dict) -> float:
        if result.get('infeasible'):
            return -1.0 if self.config.direction == 'maximize' else 1e9
        return result.get('total_deposited_keV', 0.0)


class MaterialDepositionObjective(ObjectiveFunction):
    def evaluate(self, result: dict) -> float:
        if result.get('infeasible'):
            return -1.0 if self.config.direction == 'maximize' else 1e9
        target = self.config.target_material
        if target is None:
            roles = list(result.get('deposition_by_material_keV', {}).keys())
            target = roles[0] if roles else ''
        return result.get('deposition_by_material_keV', {}).get(target, 0.0)


class InteractedFractionObjective(ObjectiveFunction):
    def evaluate(self, result: dict) -> float:
        if result.get('infeasible'):
            return -1.0 if self.config.direction == 'maximize' else 1e9
        return result.get('interacted_fraction', 0.0)


class MaterialFractionObjective(ObjectiveFunction):
    def evaluate(self, result: dict) -> float:
        if result.get('infeasible'):
            return -1.0 if self.config.direction == 'maximize' else 1e9
        target = self.config.target_material
        if target is None:
            roles = list(result.get('deposition_by_material_fraction', {}).keys())
            target = roles[0] if roles else ''
        return result.get('deposition_by_material_fraction', {}).get(target, 0.0)


class FigureOfMeritObjective(ObjectiveFunction):
    def evaluate(self, result: dict) -> float:
        if result.get('infeasible'):
            return -1.0 if self.config.direction == 'maximize' else 1e9
        return result.get('figure_of_merit', 0.0)


OBJECTIVE_REGISTRY: dict[str, type[ObjectiveFunction]] = {
    'full_absorption': FullAbsorptionObjective,
    'total_deposition': TotalDepositionObjective,
    'material_deposition': MaterialDepositionObjective,
    'interacted_fraction': InteractedFractionObjective,
    'material_fraction': MaterialFractionObjective,
    'figure_of_merit': FigureOfMeritObjective,
}


def build_objective(cfg: ObjectiveConfig) -> ObjectiveFunction:
    cls = OBJECTIVE_REGISTRY.get(cfg.name)
    if cls is None:
        raise ValueError(
            f"Unknown objective '{cfg.name}'. "
            f"Available: {list(OBJECTIVE_REGISTRY.keys())}"
        )
    return cls(cfg)
