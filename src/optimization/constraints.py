from optimization.interfaces import Constraint, ConstraintConfig


class BoundsConstraint(Constraint):
    def __init__(self, param_bounds: dict[str, tuple[float, float]]):
        self.bounds = param_bounds

    def check(self, candidate: dict[str, float | str]) -> tuple[bool, list[str]]:
        issues: list[str] = []
        for name, (lo, hi) in self.bounds.items():
            val = candidate.get(name)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                if val < lo - 1e-12:
                    issues.append(f'{name}={val} < lower bound {lo}')
                if val > hi + 1e-12:
                    issues.append(f'{name}={val} > upper bound {hi}')
        return len(issues) == 0, issues


class ManufacturingWidthConstraint(Constraint):
    def __init__(
        self,
        groove: str = 'groove_width_mm',
        wall: str = 'wall_width_mm',
        matrix_edge: str = 'matrix_edge_width_mm',
        n_channels: str = 'n_channels',
        pixel_width_mm: float = 3.0,
        default_groove: float = 0.2,
        default_wall: float = 0.05,
        default_matrix_edge: float = 0.025,
        default_n_channels: int = 12,
    ):
        self.groove = groove
        self.wall = wall
        self.matrix_edge = matrix_edge
        self.n_channels = n_channels
        self.pixel_width_mm = pixel_width_mm
        self.default_groove = default_groove
        self.default_wall = default_wall
        self.default_matrix_edge = default_matrix_edge
        self.default_n_channels = default_n_channels

    def check(self, candidate: dict[str, float | str]) -> tuple[bool, list[str]]:
        issues: list[str] = []
        g = candidate.get(self.groove, self.default_groove)
        w = candidate.get(self.wall, self.default_wall)
        m = candidate.get(self.matrix_edge, self.default_matrix_edge)
        N = candidate.get(self.n_channels, self.default_n_channels)
        if isinstance(N, str):
            return False, ['n_channels must be numeric']
        N = int(N)
        if g <= 0:
            issues.append(f'groove_width ({g}) must be positive')
        if w <= 0:
            issues.append(f'wall_width ({w}) must be positive')
        if m <= 0:
            issues.append(f'matrix_edge_width ({m}) must be positive')
        if N < 1:
            issues.append(f'n_channels ({N}) must be >= 1')
        total = 2 * m + N * g + (N - 1) * w
        if total > self.pixel_width_mm * 1.05:
            issues.append(
                f'Manufacturing sum {total:.4f} exceeds pixel_width {self.pixel_width_mm} by >5%'
            )
        return len(issues) == 0, issues


class AndConstraint(Constraint):
    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints

    def check(self, candidate: dict[str, float | str]) -> tuple[bool, list[str]]:
        all_issues: list[str] = []
        for c in self.constraints:
            ok, issues = c.check(candidate)
            all_issues.extend(issues)
        return len(all_issues) == 0, all_issues


def build_constraint(cfg: ConstraintConfig) -> Constraint:
    t = cfg.type
    p = cfg.params
    if t == 'bounds':
        return BoundsConstraint(**p)
    if t == 'manufacturing_width':
        return ManufacturingWidthConstraint(**p)
    if t == 'and':
        return AndConstraint([build_constraint(cc) for cc in p.get('constraints', [])])
    raise ValueError(f"Unknown constraint type '{t}'")
