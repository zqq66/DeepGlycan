import os
import sys
import re
import pandas as pd
from sys import getsizeof
import torch
import pickle
import gzip
import numpy as np
from glob import glob
from data.BasicClass import Residual_seq, Ion, Composition
EPS = 1e-8
np.set_printoptions(threshold=sys.maxsize)

mono_composition = {
    'H': Composition('C6H12O6') - Composition('H2O'),
    'N': Composition('C8H15O6N') - Composition('H2O'),
    'A': Composition('C11H19O9N') - Composition('H2O'),
    'G': Composition('C11H19O10N') - Composition('H2O'),
    'F': Composition('C6H12O5') - Composition('H2O'),
    'X': Composition('C5H10O5') - Composition('H2O'),
}
id2mass = {k: v.mass for k, v in mono_composition.items()}
def read_mgf(file_path):
    raw_mgf_blocks = {}
    for file in glob(os.path.join(file_path,'*mgf')):
        print('file', file)
        with open(file) as f:
            for line in f:
                if line.startswith('BEGIN IONS'):
                    product_ions_moverz = []
                    product_ions_intensity = []
                elif line.startswith('PEPMASS'):
                    mz = float(re.split(r'=|\r|\n|\s', line)[1])
                elif line.startswith('CHARGE'):
                    z = int(re.search(r'CHARGE=(\d+)\+', line).group(1))
                elif line.startswith('TITLE'):
                    scan_pattern = r'scan=(\d+)'
                    scan = re.search(scan_pattern, line)
                    scan = scan.group(1)
                elif line[0].isnumeric():
                    product_ion_moverz, product_ion_intensity = line.strip().split(' ')
                    product_ions_moverz.append(float(product_ion_moverz))
                    product_ions_intensity.append(float(product_ion_intensity))
                elif line.startswith('END IONS'):
                    rawfile = file.split('.')[0].split('/')[-1] + '.raw'
                    raw_mgf_blocks[rawfile+scan] = {'product_ions_moverz':np.array(product_ions_moverz),
                                                 'product_ions_intensity':np.array(product_ions_intensity)}

    with open(os.path.join(file_path, 'all_mgf.pkl'), 'wb') as f:
        pickle.dump(raw_mgf_blocks, f)
    return raw_mgf_blocks


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

class GraphGenerator:
    def __init__(self,
                 peptide_mass_dic,isos,
                 local_sliding_window=50, 
                 data_acquisition_upper_limit=3500,
                 mass_error_da=0.02, 
                 mass_error_ppm=10):
        self.mass_error_da = mass_error_da
        self.mass_error_ppm = mass_error_ppm
        self.data_acquisition_upper_limit = data_acquisition_upper_limit
        self.peak_feature_generation = PeakFeatureGeneration(local_sliding_window,data_acquisition_upper_limit)
        self.label_ratio = []
        self.o_mass = Composition('O').mass
        self.C_mass = 1.0034
        self.isos =  [int(i) for i in isos.split(',')]
        # self.peptide_mass_dic = peptide_mass_dic
        self.pep_mass = list(peptide_mass_dic.values())
        self.pep = list(peptide_mass_dic.keys())
        
    def __call__(self, scan, product_ions_moverz, product_ions_intensity, precursor_ion_mass, charge, mode):
        peak_feature = self.peak_feature_generation(product_ions_moverz, product_ions_intensity) 
        result_lst = []
        for iso in self.isos:
            for i, ps in enumerate(self.pep_mass):       
                node_mass, node_feat, node_sourceion = self.feature_generator(precursor_ion_mass+iso*self.C_mass, product_ions_moverz, peak_feature, charge, ps,mode)
                if np.isnan(node_feat).any() or np.isnan(node_sourceion).any():
                    print(scan)
                node_input = {'node_feat': torch.Tensor(node_feat),
                            'node_sourceion': torch.IntTensor(node_sourceion)}
            # print(node_mass)
                result_lst.append({'scan':scan,
                        'node_mass': node_mass,
                        'node_input': node_input,
                        'charge': charge,
                        'precursor_mass': precursor_ion_mass+iso*self.C_mass,
                        'isotope_shift': iso,
                        'pep_mass': ps,
                        'pep': self.pep[i]} )
        return result_lst

    def _norm_2d_along_first_dim_and_broadcast(self, array):
        """Equivalent to `linalg.norm(array, axis=0)[None, :] * ones_like(array)`."""
        output = np.zeros(array.shape, dtype=array.dtype)
        for i in np.arange(array.shape[-1]):
            output[:, i] = np.linalg.norm(array[:, i])
        return output

    def _max_2d_along_first_dim_and_broadcast(self, array):
        """Equivalent to `array.max(0)[None, :] * ones_like(array)`."""
        output = np.zeros(array.shape, dtype=array.dtype)
        for i in np.arange(array.shape[-1]):
            output[:, i] = array[:, i].max()
        return output
    
    def feature_generator(self, precursor_ion_mass, product_ions_moverz, product_ions_feature, charge, pep_mass,mode):
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
        # print('product_ions_feature', product_ions_feature.shape,'node_mass', node_mass.shape,'node_sourceion', node_sourceion.shape)
        node_mass = node_mass-pep_mass
        # print(precursor_ion_mass, pep_mass)
        glycan_ion_mask = (node_mass>0) & (node_mass<precursor_ion_mass-pep_mass+self.mass_error_da)
        node_mass = node_mass[glycan_ion_mask]
        _, indices = np.unique(node_mass, return_index=True)
        node_mass = node_mass[indices]
        node_mass_sort_idx = np.argsort(node_mass)
        node_mass = node_mass[node_mass_sort_idx]
        node_mass = np.insert(node_mass,[0,node_mass.size],[0,precursor_ion_mass-pep_mass])
        node_feature =  product_ions_feature[glycan_ion_mask][indices][node_mass_sort_idx]
        node_feature = np.pad(node_feature, pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)

        node_sourceion = node_sourceion[glycan_ion_mask][indices][node_mass_sort_idx]
        node_sourceion = np.insert(node_sourceion,[0,node_sourceion.size],[0,0])

        # print('node_mass', node_mass.shape, 'node_feature', node_feature.shape, 'node_sourceion', node_sourceion.shape)
        return node_mass, node_feature, node_sourceion
