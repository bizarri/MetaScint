from __future__ import annotations
import numpy as np
from config.schema import BenchmarkConfig, MaterialConfig
from geometry.uniform_pixel import sample_entry_xy, sample_entry_direction, distance_to_box_mm
from physics.photon_kinematics import sample_klein_nishina_cosine, rotate_direction
from physics.attenuation import attenuation_coefficients, MEC2_KEV
from physics.electron_tabata import TabataRangeModel
from physics.relaxation import post_photoelectric_secondaries


def transport_electron_csda(range_model: TabataRangeModel, pos_mm, dir_vec, E_keV, geometry):
    total_range_mm = range_model.range_mm(E_keV)
    if total_range_mm <= 0.0:
        return {'deposited_keV':0.0,'escaped_keV':E_keV,'total_range_mm':0.0,'path_inside_mm':0.0}
    dir_unit = dir_vec / np.linalg.norm(dir_vec)
    path_inside_mm = distance_to_box_mm(pos_mm, dir_unit, geometry)
    if path_inside_mm >= total_range_mm:
        return {'deposited_keV':E_keV,'escaped_keV':0.0,'total_range_mm':total_range_mm,'path_inside_mm':total_range_mm}
    E_after = range_model.residual_energy_from_path(E_keV, path_inside_mm)
    return {'deposited_keV':E_keV-E_after,'escaped_keV':E_after,'total_range_mm':total_range_mm,'path_inside_mm':path_inside_mm}


def run_uniform_material(name: str, material: MaterialConfig, cfg: BenchmarkConfig, seed_offset: int = 0, collect_per_event_data: bool = False):
    rng = np.random.default_rng(cfg.simulation.seed + seed_offset)
    gamma_E0 = cfg.simulation.gamma_energy_keV
    n_events = cfg.simulation.n_events
    cutoff = cfg.simulation.photon_energy_cutoff_keV
    geometry = cfg.geometry
    irradiation = cfg.irradiation
    range_model = TabataRangeModel(material.formula, material.density_g_cm3)
    full = partial = none = interacted = 0
    examples = []
    per_event_data = [] if collect_per_event_data else None
    for event_index in range(n_events):
        x0, y0 = sample_entry_xy(rng, geometry, irradiation)
        entry_dir = sample_entry_direction(rng, irradiation)
        dep = 0.0
        escaped_photon_energy = 0.0
        hist = []
        had_interaction = False
        branch_counter = 0
        record_history = len(examples) < cfg.simulation.n_example_histories
        photon_stack = [{'branch_id':0,'parent_branch_id':None,'pos':np.array([x0,y0,0.0],dtype=float),'dir':entry_dir,'E':gamma_E0,'kind':'primary'}]
        while photon_stack:
            photon = photon_stack.pop()
            pos = photon['pos'].copy(); photon_dir = photon['dir'].copy(); E = float(photon['E']); branch_id = photon['branch_id']
            for step in range(cfg.simulation.max_photon_steps):
                mu_total_cm1, mu_pe_cm1 = attenuation_coefficients(material.formula, material.density_g_cm3, E)
                mu_comp_cm1 = max(mu_total_cm1 - mu_pe_cm1, 0.0)
                sampled_free_path_mm = rng.exponential(scale=10.0/mu_total_cm1)
                distance_boundary_mm = distance_to_box_mm(pos, photon_dir, geometry)
                if record_history:
                    row = {'branch_id':branch_id,'step_index':step,'branch_kind':photon['kind'],'position_mm_before':pos.round(6).tolist(),'direction_before':photon_dir.round(6).tolist(),'photon_energy_keV_before':float(E),'mu_total_cm1':float(mu_total_cm1),'mu_compton_cm1':float(mu_comp_cm1),'mu_photoelectric_like_cm1':float(mu_pe_cm1),'photoelectric_fraction':float(mu_pe_cm1/mu_total_cm1 if mu_total_cm1>0 else 0.0),'sampled_free_path_mm':float(sampled_free_path_mm),'distance_to_boundary_mm':float(distance_boundary_mm)}
                else:
                    row = None
                if sampled_free_path_mm >= distance_boundary_mm:
                    pos = pos + photon_dir*distance_boundary_mm
                    escaped_photon_energy += E
                    if row is not None:
                        row['outcome'] = 'photon_escape'; row['position_mm_after'] = pos.round(6).tolist(); hist.append(row)
                    break
                pos = pos + photon_dir*sampled_free_path_mm; had_interaction = True
                if row is not None:
                    row['position_mm_after'] = pos.round(6).tolist()
                if rng.random() < (mu_pe_cm1/mu_total_cm1):
                    post = post_photoelectric_secondaries(material.dominant_photo_element, E, pos, rng)
                    if row is not None:
                        row['interaction'] = 'photoelectric_like'; row['post_photoelectric'] = post
                    if post['photoelectron_energy_keV'] is not None:
                        pe_trans = transport_electron_csda(range_model, pos, np.array(post['photoelectron_direction'],dtype=float), post['photoelectron_energy_keV'], geometry)
                        if row is not None:
                            row['post_photoelectric']['photoelectron_transport'] = pe_trans
                        dep += pe_trans['deposited_keV']
                    dep += post.get('local_relaxation_keV',0.0)
                    for sec_e in post.get('secondary_electrons', []):
                        e_trans = transport_electron_csda(range_model, pos, np.array(sec_e['direction'],dtype=float), sec_e['energy_keV'], geometry)
                        sec_e['transport'] = e_trans
                        dep += e_trans['deposited_keV']
                    for sec_p in post.get('secondary_photons', []):
                        branch_counter += 1
                        photon_stack.append({'branch_id':branch_counter,'parent_branch_id':branch_id,'pos':np.array(sec_p['position_mm'],dtype=float),'dir':np.array(sec_p['direction'],dtype=float),'E':float(sec_p['energy_keV']),'kind':'fluorescence'})
                    if row is not None:
                        hist.append(row)
                    break
                mu = sample_klein_nishina_cosine(rng, E); phi = rng.uniform(0.0,2*np.pi); scattered_dir = rotate_direction(photon_dir, mu, phi)
                E_scattered = E/(1.0 + (E/MEC2_KEV)*(1.0-mu)); recoil_e_keV = E-E_scattered
                pe_vec = E*photon_dir - E_scattered*scattered_dir; electron_dir = photon_dir.copy() if np.linalg.norm(pe_vec)<1e-12 else pe_vec/np.linalg.norm(pe_vec)
                e_trans = transport_electron_csda(range_model, pos, electron_dir, recoil_e_keV, geometry)
                dep += e_trans['deposited_keV']
                if row is not None:
                    row['interaction'] = 'compton'; row['cos_theta']=float(mu); row['phi_rad']=float(phi); row['electron_energy_keV']=float(recoil_e_keV); row['electron_direction']=electron_dir.round(6).tolist(); row['electron_transport']=e_trans; row['scattered_photon_energy_keV']=float(E_scattered); row['direction_after']=scattered_dir.round(6).tolist(); hist.append(row)
                E = E_scattered; photon_dir = scattered_dir
                if E < cutoff:
                    dep += E
                    if hist:
                        hist[-1]['below_cutoff_absorbed_locally'] = float(E)
                    break
            else:
                # Step limit reached: deposit remaining photon energy locally to conserve energy.
                dep += E
                if hist:
                    hist[-1]['step_limit_reached'] = True
                    hist[-1]['step_limit_deposited_keV'] = float(E)
        if had_interaction:
            interacted += 1
        if dep >= gamma_E0 - 1e-9:
            full += 1; cls='full'
        elif dep > 0.0:
            partial += 1; cls='partial'
        else:
            none += 1; cls='none'
        if collect_per_event_data:
            per_event_data.append({
                'deposition_by_material_keV': {name: float(dep)},
                'classification': cls,
                'deposited_keV': float(dep),
            })
        if len(examples) < cfg.simulation.n_example_histories:
            examples.append({'material':name,'event_index':event_index,'classification':cls,'deposited_keV':float(dep),'escaped_photon_energy_keV':float(escaped_photon_energy),'first_entry_xy_mm':[float(x0),float(y0)],'history':hist})
    out = {
        'material': name,
        'n_events': n_events,
        'irradiation_area_fraction': irradiation.front_face_area_fraction,
        'interacted_count': interacted,
        'interacted_fraction': interacted/n_events,
        'full_count': full,
        'full_fraction': full/n_events,
        'partial_count': partial,
        'partial_fraction': partial/n_events,
        'none_count': none,
        'none_fraction': none/n_events,
        'csda_summary': [{'energy_keV':E,'csda_range_mm':range_model.range_mm(E)} for E in [20.0,50.0,100.0,200.0,300.0,500.0]],
        'examples': examples,
    }
    if collect_per_event_data:
        out['events_data'] = per_event_data
    return out
