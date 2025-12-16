import os
import sys
import re

import torch
import pickle
import numpy as np
from glob import glob
from data.BasicClass import Residual_seq, Ion, Composition
EPS = 1e-8
np.set_printoptions(threshold=sys.maxsize)
C_mass = 1.0034
mono_composition = {
    'H': Composition('C6H12O6') - Composition('H2O'),
    'N': Composition('C8H15O6N') - Composition('H2O'),
    'A': Composition('C11H19O9N') - Composition('H2O'),
    'G': Composition('C11H19O10N') - Composition('H2O'),
    'F': Composition('C6H12O5') - Composition('H2O'),
    'X': Composition('C5H10O5') - Composition('H2O'),
}
id2mass = {k: v.mass for k, v in mono_composition.items()}

class PeakFeatureGeneration:
    def __init__(self, local_sliding_window, data_acquisition_upper_limit):
        self.local_sliding_window = local_sliding_window
        self.data_acquisition_upper_limit = data_acquisition_upper_limit
        
    def __call__(self, product_ions_moverz, product_ions_intensity):
        normalize_moverz = self.normalize_moverzCal(product_ions_moverz)
        relative_intensity = self.relative_intensityCal(product_ions_intensity)
        total_rank = self.total_rankCal(product_ions_intensity)
        total_halfrank = self.total_halfrankCal(product_ions_intensity)
        local_mask = self.local_intensity_mask(product_ions_moverz)
        local_significant = self.local_significantCal(local_mask, product_ions_intensity)
        local_rank = self.local_rankCal(local_mask,product_ions_intensity)
        local_halfrank = self.local_halfrankCal(local_mask,product_ions_intensity)
        local_reletive_intensity = self.local_reletive_intensityCal(local_mask,product_ions_intensity)

        product_ions_feature = np.stack([normalize_moverz,
                                         relative_intensity,
                                         local_significant,
                                         total_rank,
                                         total_halfrank,
                                         local_rank,
                                         local_halfrank,
                                         local_reletive_intensity]).transpose()

        return product_ions_feature
    
    def normalize_moverzCal(self, moverz):
        return np.exp(-moverz/self.data_acquisition_upper_limit)

    def relative_intensityCal(self, intensity):
        return intensity/intensity.max()

    def local_intensity_mask(self, mz):
        right_boundary = np.reshape(mz+self.local_sliding_window,(-1,1))
        left_boundary = np.reshape(mz-self.local_sliding_window,(-1,1))
        mask = np.logical_and(right_boundary>mz,left_boundary<mz)
        return mask

    def local_significantCal(self, mask, intensity): 
        local_significant=[]
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_significant.append(np.tanh((intensity[i]/local_intensity_list.min()-1)/2))
        return np.array(local_significant)

    def local_rankCal(self, mask, intensity):
        local_rank = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_rank.append(np.sum(intensity[i]>local_intensity_list)/len(local_intensity_list))
        return np.array(local_rank)

    def local_halfrankCal(self, mask, intensity):
        local_halfrank = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_halfrank.append(np.sum(intensity[i]/2>local_intensity_list)/len(local_intensity_list))
        return np.array(local_halfrank)

    def local_reletive_intensityCal(self, mask, intensity):
        local_reletive_intensity=[]
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_reletive_intensity.append(intensity[i]/local_intensity_list.max())
        return np.array(local_reletive_intensity)

    def total_rankCal(self, intensity):
        temp_intensity = intensity.reshape((-1,1))
        return np.sum(temp_intensity>intensity,axis=1)/len(intensity)

    def total_halfrankCal(self, intensity):
        half_intensity = intensity/2
        half_intensity = half_intensity.reshape((-1,1))
        return np.sum(half_intensity>intensity,axis=1)/len(intensity)

class GraphGeneratorOptimized:
    """Optimised graph generator producing node masses and features.

    This class is API‑compatible with the original :class:`GraphGenerator`
    but allows the user to choose the implementation used to compute peak
    features.  See the module docstring for details on the available
    back‑ends.
    """

    def __init__(
        self,
        local_sliding_window: float = 50.0,
        data_acquisition_upper_limit: float = 3500.0,
        mass_error_da: float = 0.02,
        mass_error_ppm: float = 10.0,
        pep_masses=None,
        isotopes = None,
        min_cand = 0

    ) -> None:
        self.mass_error_da = mass_error_da
        self.mass_error_ppm = mass_error_ppm
        self.data_acquisition_upper_limit = data_acquisition_upper_limit
        
        self.feature_generator_impl = PeakFeatureGeneration(local_sliding_window,data_acquisition_upper_limit)

        self.o_mass = Composition('O').mass
        self.pep_masses = list(pep_masses.values())
        self.pep_seqs = list(pep_masses.keys())
        self.isotopes = [int(i) for i in isotopes.split(',')]
        self.min_cand = min_cand
        mass_N = 203.079373
        mass_H = 162.052824
        self.y = [mass_N,2 * mass_N]

        for i in range(1, 3 + 1):
            self.y.append(self.y[-1] + mass_H)
        self.y = np.array(self.y)
        print(self.y)

    def __call__(
        self,
        scan: str,
        product_ions_moverz: np.ndarray,
        product_ions_intensity: np.ndarray,
        precursor_ion_mass: float,
        charge: int,
        mode: str):
        """Generate node masses and features for a single spectrum.

        Parameters
        ----------
        scan: str
            Identifier for the scan (carried through unmodified).
        product_ions_moverz: np.ndarray
            m/z values of the product ions (1D array).
        product_ions_intensity: np.ndarray
            Intensity values of the product ions (1D array, same length as
            ``product_ions_moverz``).
        precursor_ion_mass: float
            Mass of the precursor ion.
        multi_charged: bool
            If ``True``, include peaks generated at charges 1–3; otherwise
            include only charges 1–2.
        pep_mass: float
            Mass of the peptide backbone.  The node masses are offset by
            ``pep_mass`` and the final start/end nodes are inserted at
            0 and ``precursor_ion_mass - pep_mass``.
        mode: str
            Dissociation mode.  If equal to ``'ethcd'`` the algorithm
            generates additional peaks corresponding to oxygen addition.

        Returns
        -------
        node_mass : np.ndarray
            Sorted 1D array of node masses including the start and end nodes.
        node_input : dict
            Dictionary containing ``'node_feat'`` (a 2D array of features
            aligned with ``node_mass``) and ``'node_sourceion'`` (a 1D
            integer array indicating the charge state origin of each peak).
        """
        # Compute per‑peak features using the selected implementation.
        product_ions_feature = self.feature_generator_impl(product_ions_moverz, product_ions_intensity)

        if charge > 2:
            node_mass = np.concatenate([product_ions_moverz,Ion.mass2mz(product_ions_moverz, 2),Ion.mass2mz(product_ions_moverz, 3)])
            product_ions_feature = np.repeat(product_ions_feature,3,axis=0)
            node_sourceion = np.concatenate([np.ones(product_ions_moverz.shape[0]),2*np.ones(product_ions_moverz.shape[0]),3*np.ones(product_ions_moverz.shape[0])])
        else:
            node_mass = np.concatenate([product_ions_moverz,Ion.mass2mz(product_ions_moverz, 2)])
            product_ions_feature = np.repeat(product_ions_feature,2,axis=0)
            node_sourceion = np.concatenate([np.ones(product_ions_moverz.shape[0]),2*np.ones(product_ions_moverz.shape[0])])
        if mode == 'ethcd':
            node_mass = np.concatenate([node_mass, node_mass + self.o_mass])
            product_ions_feature = np.repeat(product_ions_feature,2,axis=0)
            node_sourceion_z = node_sourceion.copy()*2
            node_sourceion = np.concatenate([node_sourceion, node_sourceion_z])

        results = self.obtain_glycan_mass_feature(node_mass, product_ions_feature, node_sourceion,
                                        self.pep_masses, precursor_ion_mass)
        return [{'scan':scan,
                'node_mass': node_mass,
                'node_input': {'node_feat':node_feat,'node_sourceion':node_ion},
                'charge': charge,
                'precursor_mass': precursor_ion_mass+iso*C_mass,
                'isotope_shift': iso,
                'pep_mass': self.pep_masses[i],
                'pep': self.pep_seqs[i]} for (i, iso, node_mass, node_feat, node_ion) in results]
            
    def ppm_window(self,theo: np.ndarray, tol_ppm: float) -> np.ndarray:
        delta = theo * tol_ppm * 1e-6
        return np.column_stack((theo - delta, theo + delta))

    def match_ladder_ions(self,mass_list,
            tol_ppm: float = 20.0,
            max_hex: int = 5):
        win   = self.ppm_window(self.y, tol_ppm)                  # (m, 2)
        # print(mass_list.shape)
        # Broadcasting trick: compare each peak with each ion window
        #   mass_matrix[..., None]  -> (n, l, 1)
        #   win[None, None, :, 0/1] -> (1, 1, m)
        lower_ok = mass_list[..., None] >= win[None, None, :, 0]
        upper_ok = mass_list[..., None] <= win[None, None, :, 1]
        hit_any  = np.any(lower_ok & upper_ok, axis=1)            # collapse l  ⇒ (n, m)
        return hit_any


    def obtain_glycan_mass_feature(self,node_mass_raw,node_feat_raw,node_sourceion_raw,
                               pep_masses, precursor_mass, mass_error_da=0.02):
        pep_masses_arr = np.asarray(pep_masses).reshape(-1)
        adjusted_masses = node_mass_raw[:, None] - pep_masses_arr[None, :]
        matched_ladder_ions = self.match_ladder_ions(np.transpose(adjusted_masses))
        per_ion_counts =  matched_ladder_ions.sum(axis=1)
        # print("Per ion counts:", per_ion_counts)
        results = []
        for k, pep_mass in enumerate(pep_masses):
            # if per_ion_counts[k] <2:
            #     continue
            node_mass = adjusted_masses[:,k]
            max_glycan_mass = precursor_mass - pep_mass + mass_error_da
            glycan_ion_mask = (node_mass > 0.0) & (node_mass < max_glycan_mass)
            node_mass = node_mass[glycan_ion_mask]
            _, indices = np.unique(node_mass, return_index=True)
            node_mass = node_mass[indices]
            node_mass_sort_idx = np.argsort(node_mass)
            node_mass = node_mass[node_mass_sort_idx]
            node_feature =  node_feat_raw[glycan_ion_mask][indices][node_mass_sort_idx]
            node_feature = np.pad(node_feature, pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)

            node_sourceion = node_sourceion_raw[glycan_ion_mask][indices][node_mass_sort_idx]
            node_sourceion = np.insert(node_sourceion,[0,node_sourceion.size],[0,0])

            for iso in self.isotopes:
                if precursor_mass+iso*C_mass - pep_mass > self.min_cand:
                    node_mass_iso = np.insert(node_mass,[0,node_mass.size],[0,precursor_mass+iso*C_mass-pep_mass])
                    results.append((k, iso, node_mass_iso, node_feature, node_sourceion))
        return results