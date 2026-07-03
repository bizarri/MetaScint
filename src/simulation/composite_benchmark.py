from __future__ import annotations
import numpy as np
from config.schema import BenchmarkConfig, MaterialConfig
from geometry.composite_pixel import CompositeGeometry
from physics.photon_kinematics import sample_klein_nishina_cosine, rotate_direction
from physics.attenuation import attenuation_coefficients, MEC2_KEV
from physics.electron_tabata import TabataRangeModel
from physics.relaxation import post_photoelectric_secondaries


def transport_electron_csda(
    range_models: dict[str, TabataRangeModel],
    pos_mm, dir_vec, E_keV, geometry: CompositeGeometry,
):
    dir_unit = dir_vec / np.linalg.norm(dir_vec)
    current_pos = np.array(pos_mm, dtype=float)
    rem_E = float(E_keV)
    dep_by_mat: dict[str, float] = {}
    total_path = 0.0

    while rem_E > 0.0:
        mat = geometry.material_role_at(current_pos)
        rm = range_models[mat]
        R_current = rm.range_mm(rem_E)
        if R_current <= 0.0:
            break

        dist_to_change, next_mat = geometry.distance_to_next_material_change(current_pos, dir_unit)
        box_dist = geometry.distance_to_box_mm(current_pos, dir_unit)

        if dist_to_change >= box_dist - 1e-12:
            path_here = min(box_dist, R_current)
            if path_here >= R_current - 1e-12:
                dep_by_mat[mat] = dep_by_mat.get(mat, 0.0) + rem_E
                total_path += R_current
                return {
                    'deposition_by_material_keV': dep_by_mat,
                    'deposited_keV': sum(dep_by_mat.values()),
                    'escaped_keV': 0.0,
                    'total_range_mm': R_current,
                    'path_inside_mm': total_path,
                }
            E_after = rm.residual_energy_from_path(rem_E, path_here)
            dep_by_mat[mat] = dep_by_mat.get(mat, 0.0) + rem_E - E_after
            current_pos = current_pos + dir_unit * path_here
            total_path += path_here
            return {
                'deposition_by_material_keV': dep_by_mat,
                'deposited_keV': sum(dep_by_mat.values()),
                'escaped_keV': E_after,
                'total_range_mm': None,
                'path_inside_mm': total_path,
            }

        path_here = min(dist_to_change, R_current)
        if path_here >= R_current - 1e-12:
            dep_by_mat[mat] = dep_by_mat.get(mat, 0.0) + rem_E
            total_path += R_current
            return {
                'deposition_by_material_keV': dep_by_mat,
                'deposited_keV': sum(dep_by_mat.values()),
                'escaped_keV': 0.0,
                'total_range_mm': R_current,
                'path_inside_mm': total_path,
            }

        E_after = rm.residual_energy_from_path(rem_E, path_here)
        dep_by_mat[mat] = dep_by_mat.get(mat, 0.0) + rem_E - E_after
        current_pos = current_pos + dir_unit * path_here
        total_path += path_here
        rem_E = E_after

    return {
        'deposition_by_material_keV': dep_by_mat,
        'deposited_keV': sum(dep_by_mat.values()),
        'escaped_keV': rem_E,
        'total_range_mm': None,
        'path_inside_mm': total_path,
    }


def run_composite_benchmark(
    materials: dict[str, MaterialConfig],
    geometry: CompositeGeometry,
    cfg: BenchmarkConfig,
    seed_offset: int = 0,
    collect_per_event_data: bool = False,
):
    rng = np.random.default_rng(cfg.simulation.seed + seed_offset)
    gamma_E0 = cfg.simulation.gamma_energy_keV
    n_events = cfg.simulation.n_events
    cutoff = cfg.simulation.photon_energy_cutoff_keV
    irradiation = cfg.irradiation

    range_models = {
        role: TabataRangeModel(mat.formula, mat.density_g_cm3)
        for role, mat in materials.items()
    }

    material_roles = list(materials.keys())
    dep_sum_by_mat = {role: 0.0 for role in material_roles}
    full = partial = none = interacted = 0
    examples = []
    per_event_data = [] if collect_per_event_data else None

    for event_index in range(n_events):
        x0, y0 = geometry.sample_entry_xy(rng, irradiation)
        entry_dir = geometry.sample_entry_direction(rng, irradiation)
        dep_by_mat = {role: 0.0 for role in material_roles}
        escaped_photon_energy = 0.0
        hist = []
        had_interaction = False
        branch_counter = 0
        record_history = len(examples) < cfg.simulation.n_example_histories

        photon_stack = [{
            'branch_id': 0, 'parent_branch_id': None,
            'pos': np.array([x0, y0, 0.0], dtype=float),
            'dir': entry_dir,
            'E': gamma_E0, 'kind': 'primary',
        }]

        while photon_stack:
            photon = photon_stack.pop()
            pos = photon['pos'].copy()
            photon_dir = photon['dir'].copy()
            E = float(photon['E'])
            branch_id = photon['branch_id']

            remaining_path_integral = None

            for step in range(cfg.simulation.max_photon_steps):
                mat_role = geometry.material_role_at(pos)
                material = materials[mat_role]
                range_model = range_models[mat_role]

                mu_total_cm1, mu_pe_cm1 = attenuation_coefficients(material.formula, material.density_g_cm3, E)
                mu_comp_cm1 = max(mu_total_cm1 - mu_pe_cm1, 0.0)

                if remaining_path_integral is None:
                    remaining_path_integral = -np.log(rng.random())

                dist_to_change, next_mat = geometry.distance_to_next_material_change(pos, photon_dir)
                delta_cm = mu_total_cm1 * (dist_to_change / 10.0)

                row = {
                    'branch_id': branch_id, 'step_index': step,
                    'branch_kind': photon['kind'],
                    'position_mm_before': pos.round(6).tolist(),
                    'direction_before': photon_dir.round(6).tolist(),
                    'photon_energy_keV_before': float(E),
                    'material_role': mat_role,
                    'mu_total_cm1': float(mu_total_cm1),
                    'mu_compton_cm1': float(mu_comp_cm1),
                    'mu_photoelectric_like_cm1': float(mu_pe_cm1),
                    'photoelectric_fraction': float(mu_pe_cm1 / mu_total_cm1 if mu_total_cm1 > 0 else 0.0),
                    'remaining_path_integral': float(remaining_path_integral),
                    'distance_to_material_boundary_mm': float(dist_to_change),
                } if record_history else None

                if remaining_path_integral > delta_cm:
                    pos = pos + photon_dir * dist_to_change
                    remaining_path_integral -= delta_cm
                    if next_mat is None:
                        escaped_photon_energy += E
                        if row is not None:
                            row['outcome'] = 'photon_escape'
                            row['position_mm_after'] = pos.round(6).tolist()
                            hist.append(row)
                        break
                    else:
                        if row is not None:
                            row['outcome'] = 'material_transition'
                            row['next_material'] = next_mat
                            row['position_mm_after'] = pos.round(6).tolist()
                            hist.append(row)
                        continue

                interact_dist_mm = (remaining_path_integral / mu_total_cm1) * 10.0
                pos = pos + photon_dir * interact_dist_mm
                remaining_path_integral = None
                had_interaction = True
                if row is not None:
                    row['position_mm_after'] = pos.round(6).tolist()
                    row['interaction_distance_mm'] = float(interact_dist_mm)

                if rng.random() < (mu_pe_cm1 / mu_total_cm1):
                    post = post_photoelectric_secondaries(material.dominant_photo_element, E, pos, rng)
                    if row is not None:
                        row['interaction'] = 'photoelectric_like'
                        row['post_photoelectric'] = post

                    if post['photoelectron_energy_keV'] is not None:
                        pe_trans = transport_electron_csda(
                            range_models, pos,
                            np.array(post['photoelectron_direction'], dtype=float),
                            post['photoelectron_energy_keV'], geometry,
                        )
                        if row is not None:
                            row['post_photoelectric']['photoelectron_transport'] = pe_trans
                        for role, val in pe_trans['deposition_by_material_keV'].items():
                            dep_by_mat[role] = dep_by_mat.get(role, 0.0) + val

                    dep_by_mat[mat_role] += post.get('local_relaxation_keV', 0.0)
                    for sec_e in post.get('secondary_electrons', []):
                        e_trans = transport_electron_csda(
                            range_models, pos,
                            np.array(sec_e['direction'], dtype=float),
                            sec_e['energy_keV'], geometry,
                        )
                        sec_e['transport'] = e_trans
                        for role, val in e_trans['deposition_by_material_keV'].items():
                            dep_by_mat[role] = dep_by_mat.get(role, 0.0) + val

                    for sec_p in post.get('secondary_photons', []):
                        branch_counter += 1
                        photon_stack.append({
                            'branch_id': branch_counter, 'parent_branch_id': branch_id,
                            'pos': np.array(sec_p['position_mm'], dtype=float),
                            'dir': np.array(sec_p['direction'], dtype=float),
                            'E': float(sec_p['energy_keV']),
                            'kind': 'fluorescence',
                        })

                    if row is not None:
                        hist.append(row)
                    break

                mu = sample_klein_nishina_cosine(rng, E)
                phi = rng.uniform(0.0, 2 * np.pi)
                scattered_dir = rotate_direction(photon_dir, mu, phi)
                E_scattered = E / (1.0 + (E / MEC2_KEV) * (1.0 - mu))
                recoil_e_keV = E - E_scattered

                pe_vec = E * photon_dir - E_scattered * scattered_dir
                electron_dir = photon_dir.copy() if np.linalg.norm(pe_vec) < 1e-12 else pe_vec / np.linalg.norm(pe_vec)

                e_trans = transport_electron_csda(range_models, pos, electron_dir, recoil_e_keV, geometry)
                for role, val in e_trans['deposition_by_material_keV'].items():
                    dep_by_mat[role] = dep_by_mat.get(role, 0.0) + val

                if row is not None:
                    row['interaction'] = 'compton'
                    row['cos_theta'] = float(mu)
                    row['phi_rad'] = float(phi)
                    row['electron_energy_keV'] = float(recoil_e_keV)
                    row['electron_direction'] = electron_dir.round(6).tolist()
                    row['electron_transport'] = e_trans
                    row['scattered_photon_energy_keV'] = float(E_scattered)
                    row['direction_after'] = scattered_dir.round(6).tolist()
                    hist.append(row)

                E = E_scattered
                photon_dir = scattered_dir

                if E < cutoff:
                    dep_by_mat[mat_role] += E
                    if hist:
                        hist[-1]['below_cutoff_absorbed_locally'] = float(E)
                    break

            else:
                # Step limit reached: re-query the current material at `pos` rather
                # than using `mat_role`, which may be stale after a material_transition
                # step was the final iteration. Deposit remaining energy there.
                current_mat_role = geometry.material_role_at(pos)
                dep_by_mat[current_mat_role] = dep_by_mat.get(current_mat_role, 0.0) + E
                if hist:
                    hist[-1]['step_limit_reached'] = True
                    hist[-1]['step_limit_deposited_keV'] = float(E)

        event_dep = sum(dep_by_mat.values())
        # Accumulate per-event deposits into the run total. Use dep_by_mat.items()
        # rather than material_roles so that any role returned by transport_electron_csda
        # (e.g. default_material_role when it differs from the named material roles) is
        # not silently dropped.
        for role, val in dep_by_mat.items():
            dep_sum_by_mat[role] = dep_sum_by_mat.get(role, 0.0) + val

        if event_dep >= gamma_E0 - 1e-9:
            full += 1
            cls = 'full'
        elif event_dep > 0.0:
            partial += 1
            cls = 'partial'
        else:
            none += 1
            cls = 'none'

        if collect_per_event_data:
            per_event_data.append({
                'deposition_by_material_keV': {role: float(v) for role, v in dep_by_mat.items()},
                'classification': cls,
                'deposited_keV': float(event_dep),
            })

        if had_interaction:
            interacted += 1

        if len(examples) < cfg.simulation.n_example_histories:
            examples.append({
                'event_index': event_index,
                'classification': cls,
                'deposited_keV': float(event_dep),
                'deposited_by_material_keV': {role: float(v) for role, v in dep_by_mat.items()},
                'escaped_photon_energy_keV': float(escaped_photon_energy),
                'first_entry_xy_mm': [float(x0), float(y0)],
                'history': hist,
            })

    total_dep = sum(dep_sum_by_mat.values())

    out = {
        'n_events': n_events,
        'irradiation_area_fraction': irradiation.front_face_area_fraction,
        'interacted_count': interacted,
        'interacted_fraction': interacted / n_events,
        'full_count': full,
        'full_fraction': full / n_events,
        'partial_count': partial,
        'partial_fraction': partial / n_events,
        'none_count': none,
        'none_fraction': none / n_events,
        'total_deposited_keV': float(total_dep),
        'deposition_by_material_keV': {role: float(v) for role, v in dep_sum_by_mat.items()},
        'deposition_by_material_fraction': {role: float(dep_sum_by_mat[role] / (n_events * gamma_E0)) for role in material_roles},
        'examples': examples,
    }
    if collect_per_event_data:
        out['events_data'] = per_event_data
    return out
