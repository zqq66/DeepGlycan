import os
import sys
import re
import gzip
import json
import copy
import torch
import pickle
import numpy as np
import pandas as pd
import more_itertools
import collections
from glob import glob
import itertools as it
import math
from sys import getsizeof
from BasicClass import Residual_seq, Ion, Composition
from itertools import combinations_with_replacement
from edge_matrix_gen import typological_sort_floyd_warshall
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

EPS = 1e-8

all_edge_mass = []
aalist = Residual_seq.output_aalist()[-5:]
for num in range(1,3):
    for i in combinations_with_replacement(aalist,num):
        all_edge_mass.append(Residual_seq(i).mass)
all_edge_mass = np.unique(np.rint(all_edge_mass).astype(np.int64))

with open('candidate_mass','rb') as f:
    candidate_mass = pickle.load(f)

mono_composition = {
    'H': Composition('C6H12O6') - Composition('H2O'),
    'N': Composition('C8H15O6N') - Composition('H2O'),
    'A': Composition('C11H19O9N') - Composition('H2O'),
    'G': Composition('C11H19O10N') - Composition('H2O'),
    'F': Composition('C6H12O5') - Composition('H2O'),
}
diag_ion=np.array([204.086646, 186.076086, 168.065526, 366.139466, 144.0656, 138.055, 512.197375, 292.1026925, 274.0921325,
          657.2349, 243.026426, 405.079246, 485.045576, 308.09761])
id2mass = {k: v.mass for k, v in mono_composition.items()}
def read_mgf(file_path):
    raw_mgf_blocks = {}
    all_pre = {}
    for file in glob(os.path.join(file_path,'*mgf')):
        print('file', file)
        with open(file) as f:
            for line in f:
                if line.startswith('BEGIN IONS'):
                    product_ions_moverz = []
                    product_ions_intensity = []
                elif line.startswith('PEPMASS'):
                    mz = float(re.split('=|\r|\n|\s', line)[1])
                elif line.startswith('CHARGE'):
                    z = int(re.search(r'CHARGE=(\d+)\+', line).group(1))
                elif line.startswith('TITLE'):
                    scan_pattern = r'scan=(\d+)'
                    scan = re.search(scan_pattern, line)
                    scan = scan.group(1)
                    
                elif line[0].isnumeric():
                    product_ion_moverz, product_ion_intensity = line.strip().split(' ')
                    product_ions_moverz.append(float(product_ion_moverz))
                    product_ions_intensity.append(int(float(product_ion_intensity)))
                elif line.startswith('END IONS'):
                    moverzs =np.array(product_ions_moverz)
                    start_index = moverzs.searchsorted(diag_ion - 0.02)
                    end_index = moverzs.searchsorted(diag_ion + 0.02)
                    diag_int = np.concatenate([product_ions_intensity[start:end] for start, end in zip(start_index, end_index)])
                    # print(diag_int, diag_int>1e5, 1e5)
                    if np.any(diag_int>1e5): #and np.any(abs(moverzs-308.098) < 0.02):# np.any(end_index > start_index): #
                        rawfile = file.split('.')[0].split('/')[-1] + '.raw'
                        raw_mgf_blocks[rawfile + scan] = {'product_ions_moverz': np.array(product_ions_moverz),
                                                        'product_ions_intensity': np.array(product_ions_intensity)}
                        all_pre[rawfile + scan] = {'precursor_mass': mz * z - Composition('proton').mass * (z - 1),
                                                'precursor_charge': z}
    with open(os.path.join(file_path, 'filtered_mgf.pkl'), 'wb') as f:
        pickle.dump(raw_mgf_blocks, f)
    with open(os.path.join(file_path, 'filtered_precursor.pkl'), 'wb') as f:
        pickle.dump(all_pre, f)
    return raw_mgf_blocks, all_pre

def within_tol_kdtree(A, B, eps):
    A = np.asarray(A).reshape(-1,1)
    B = np.asarray(B).reshape(-1,1)
    tree = cKDTree(B)            # O(M log M)
    # for each point in A find the distance to its nearest neighbor in B
    dists, _ = tree.query(A, k=1)  # O(N log M)
    return dists <= eps

class PepMass:
    def __init__(self, poss_pep, poss_pep_mass):
        self.theo_mass = [self.find_theo_pep_mass(pep) for pep in poss_pep]
        self.poss_pep_mass = poss_pep_mass
        self.poss_pep = poss_pep
    def find_theo_pep_mass(self, pep):
        pep_theo_mass = np.array([Ion.peptide2ionmz(pep[:i],'b',1) for i in range(1,len(pep)+1)])
        pep_theo_mass = np.concatenate((pep_theo_mass, np.array([Ion.peptide2ionmz(pep[:i],'a',1) for i in range(1,len(pep)+1)])),axis=0)
        pep_theo_mass = np.concatenate((pep_theo_mass, np.array([Ion.peptide2ionmz(pep[:i],'b',2) for i in range(1,len(pep)+1)])),axis=0)
        pep = pep[::-1]
        pep_theo_mass = np.concatenate((pep_theo_mass, np.array([Ion.peptide2ionmz(pep[:i],'y',1) for i in range(1,len(pep)+1)])),axis=0)
        
        return pep_theo_mass
    def find_match(self,spec_index, threshold=0.02):
        available_mass = []
        product_ions_moverz = all_spectra[spec_index]['product_ions_moverz']
        precursor_ion_mass = all_pre[spec_index]['precursor_mass']
        for i, theo_mass in enumerate(self.theo_mass):
            glycan_mass = precursor_ion_mass - self.poss_pep_mass[i]
            if glycan_mass > 4000 or glycan_mass < 0:
                continue
            glycan_masses = np.array([glycan_mass, glycan_mass-1.0033548378, glycan_mass+1.0033548378])
            glycan_match = within_tol_kdtree(glycan_masses, masses_array, threshold)
            if not np.any(glycan_match):
                matches = within_tol_kdtree(theo_mass, product_ions_moverz, threshold).reshape(4,-1)
                matches = np.any(matches, axis=0)
                if np.sum(matches) >= math.floor(0.5 * (len(self.poss_pep[i]))):
                    available_mass.append((glycan_mass, self.poss_pep_mass[i],self.poss_pep[i],spec_index))
        return available_mass
    
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

    def local_significantCal(self, mask, intensity): #This feature need to be fixed use signal to ratio to replace intensity.
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
                 candidate_mass,
                 theo_edge_mass,
                 local_sliding_window=50, 
                 data_acquisition_upper_limit=3500,
                 mass_error_da=0.02, 
                 mass_error_ppm=10):
        self.mass_error_da = mass_error_da
        self.mass_error_ppm = mass_error_ppm
        self.theo_edge_mass = theo_edge_mass
        self.candidate_mass = candidate_mass
        self.data_acquisition_upper_limit = data_acquisition_upper_limit
        self.peak_feature_generation = PeakFeatureGeneration(local_sliding_window,data_acquisition_upper_limit)
        # self.n_term_ion_list = ['1a','1a-NH3','1a-H2O','1b','1b-NH3','1b-H2O','2a','2a-NH3','2a-H2O','2b','2b-NH3','2b-H2O']
        self.c_term_ion_list = ['1y','1y-H2O', '1y-H2O2','2y','2y-H2O','2y-H2O2', '3y', '3y-H2O']
        self.label_ratio = []
        
    def __call__(self, scan, product_ions_moverz, product_ions_intensity, precursor_ion_mass, muti_charged,mode, glycan_data=False, pep_mass=None, decoy=False):
        peak_feature = self.peak_feature_generation(product_ions_moverz, product_ions_intensity)
        subnode_mass, subnode_feature = self.candidate_subgraph_generator(precursor_ion_mass, product_ions_moverz, peak_feature)
        if glycan_data:
            # pep_mass = Ion.peptide2mz(seq, 1)

            node_mass = self.graphnode_mass_generator(precursor_ion_mass, product_ions_moverz, muti_charged,mode, pep_mass)
            edge_matrix, edge_mask = self.edge_generator(node_mass, pep_mass)

        else:
            node_mass = self.graphnode_mass_generator(precursor_ion_mass, product_ions_moverz, muti_charged,mode)
            # subedge_maxnum, edge_type, edge_error = self.edge_generator(node_mass, precursor_ion_mass)
            edge_matrix, edge_mask = self.edge_generator(node_mass)

        #assert node_mass.size<=512
        # all neighbourhood edge are counted as node feature
        node_feat, node_sourceion = self.graphnode_feature_generator(node_mass, subnode_mass, subnode_feature, precursor_ion_mass)

        dist, predecessors = typological_sort_floyd_warshall(edge_mask)
        if np.isnan(node_feat).any() or np.isnan(node_sourceion).any():
            print(scan)
        node_input = {'node_feat': torch.Tensor(node_feat),
                      'node_sourceion': torch.IntTensor(node_sourceion)}
        edge_input = {'edge_feat': torch.Tensor(edge_matrix),
                      'adj_matrix': torch.IntTensor(edge_mask)}
        rel_input = {'dist': torch.Tensor(dist),
                     'predecessors': torch.Tensor(predecessors),}

        return node_mass, node_input, rel_input, edge_input

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

    def candidate_subgraph_generator(self, precursor_ion_mass, product_ions_moverz, product_ions_feature):
        candidate_subgraphnode_moverz = []
        # candidate_subgraphnode_moverz += [Ion.peak2sequencemz(product_ions_moverz,ion) for ion in self.n_term_ion_list]
        candidate_subgraphnode_moverz += [Ion.peak2sequencemz(product_ions_moverz,ion) for ion in self.c_term_ion_list]
        candidate_subgraphnode_moverz = np.concatenate(candidate_subgraphnode_moverz)
        candidate_subgraphnode_feature = []
        for i in range(2,len(self.c_term_ion_list)+2):
            candidate_subgraphnode_source = i*np.ones([product_ions_moverz.size, 1])
            candidate_subgraphnode_feature.append(np.concatenate((product_ions_feature,candidate_subgraphnode_source),axis=1))
        candidate_subgraphnode_feature = np.concatenate(candidate_subgraphnode_feature)
        
        candidate_subgraphnode_moverz = np.insert(candidate_subgraphnode_moverz,
                                                  [0,candidate_subgraphnode_moverz.size],
                                                  [0,precursor_ion_mass])
        sorted_index = np.argsort(candidate_subgraphnode_moverz)
        candidate_subgraphnode_moverz = candidate_subgraphnode_moverz[sorted_index]
        
        candidate_subgraphnode_feature = np.concatenate([np.array([1]*9).reshape(1,-1),
                                                         candidate_subgraphnode_feature,
                                                         np.array([1]*9).reshape(1,-1)],axis=0)
        candidate_subgraphnode_feature = candidate_subgraphnode_feature[sorted_index]
        return candidate_subgraphnode_moverz, candidate_subgraphnode_feature

    def record_filter(self, mass_list, precursor_ion_mass=None, pep_mass=None):
        if precursor_ion_mass:
            mass_threshold = self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        else:
            mass_threshold = self.mass_error_da
        if pep_mass:
            mask = self.candidate_mass.searchsorted(mass_list -pep_mass - mass_threshold) != self.candidate_mass.searchsorted(
                mass_list + mass_threshold)
            mask = np.logical_or(mask, mass_list >= self.candidate_mass.max())
            mask = np.logical_and(mask, mass_list >= pep_mass)
        else:
            mask = self.candidate_mass.searchsorted(mass_list - mass_threshold) != self.candidate_mass.searchsorted(
                mass_list + mass_threshold)
            mask = np.logical_or(mask, mass_list >= self.candidate_mass.max())
        if precursor_ion_mass:
            mask = np.logical_and(mask, mass_list < precursor_ion_mass)
        return mass_list[mask], mask

    def graphnode_mass_generator(self, precursor_ion_mass, product_ions_moverz, muti_charged, mode,pep_mass=None):
        # for ethcd we consider both glycan y and z ions, while for scehcd only glycan-y ions under consideration
        node_1y_mass_cterm, _ = self.record_filter(product_ions_moverz, precursor_ion_mass, pep_mass)
        # node_1y_mass_nterm, _ = self.record_filter(precursor_ion_mass-node_1y_mass_cterm,precursor_ion_mass)
        # 2y ion
        node_2y_mass_cterm, _ = self.record_filter(Ion.mass2mz(product_ions_moverz, 2), precursor_ion_mass, pep_mass)
        node_3y_mass_cterm, _ = self.record_filter(
            Ion.mass2mz(product_ions_moverz, 3), precursor_ion_mass, pep_mass)
        if mode == 'ethcd':
            node_1z_mass_cterm, _ = self.record_filter(product_ions_moverz + Composition('O').mass, precursor_ion_mass,
                                                       pep_mass)
            node_2z_mass_cterm, _ = self.record_filter(node_2y_mass_cterm + Composition('O').mass, precursor_ion_mass,
                                                       pep_mass)
            node_3z_mass_cterm, _ = self.record_filter(node_3y_mass_cterm + Composition('O').mass, precursor_ion_mass,
                                                       pep_mass)

        if muti_charged:
            if mode == 'ethcd':
                graphnode_mass = np.concatenate(
                    [node_1y_mass_cterm, node_1z_mass_cterm, node_2y_mass_cterm, node_2z_mass_cterm, node_3y_mass_cterm,
                     node_3z_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
            elif mode == 'scehcd':
                graphnode_mass = np.concatenate(
                    [node_1y_mass_cterm, node_2y_mass_cterm,
                     node_3y_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
            else:
                raise NotImplementedError
        else:
            if mode == 'ethcd':
                graphnode_mass = np.concatenate(
                    [node_1y_mass_cterm, node_1z_mass_cterm, node_2y_mass_cterm,
                     node_2z_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
            elif mode == 'scehcd':
                graphnode_mass = np.concatenate(
                    [node_1y_mass_cterm, node_2y_mass_cterm, node_3y_mass_cterm])
            else:
                raise NotImplementedError
        graphnode_mass = np.unique(graphnode_mass-(pep_mass-self.mass_error_da))
        graphnode_mass = np.insert(graphnode_mass,
                                   [0,graphnode_mass.size],
                                   [0,precursor_ion_mass-pep_mass ])
        return graphnode_mass

    def graphnode_feature_generator(self, graphnode_mass, subnode_mass, subnode_feature, precursor_ion_mass):
        mass_threshold = 2*self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        lower_bounds = subnode_mass.searchsorted(graphnode_mass - mass_threshold)
        higher_bounds = subnode_mass.searchsorted(graphnode_mass + mass_threshold)
        subnode_maxnum = (higher_bounds-lower_bounds).max()
        node_feature = []
        for i,(lower_bound,higher_bound) in enumerate(zip(lower_bounds,higher_bounds)):
            mass_merge_error = np.abs(graphnode_mass[i] - subnode_mass[np.arange(lower_bound,higher_bound)])
            mass_merge_index = np.argsort(mass_merge_error)
            mass_merge_error = np.exp(-np.abs(mass_merge_error)/mass_threshold)
            mass_merge_feat = np.concatenate([np.exp(-graphnode_mass[i]*np.ones((higher_bound-lower_bound,1))/self.data_acquisition_upper_limit),
                                              subnode_feature[np.arange(lower_bound,higher_bound)],
                                              mass_merge_error.reshape(-1,1)],axis=1)
            mass_merge_feat = mass_merge_feat[mass_merge_index]
            mass_merge_feat = np.pad(mass_merge_feat,((0,subnode_maxnum-(higher_bound-lower_bound)),(0,0)))
            node_feature.append(mass_merge_feat)
        node_feature = np.stack(node_feature)
        node_feature[0,1:,:]=0
        node_sourceion = node_feature[:,:,-2]
        node_feat = np.delete(node_feature,-2,axis=2)
        return node_feat, node_sourceion

    def edge_generator(self, graphnode_moverz, pepmass=None):
        n = graphnode_moverz.size
        mass_threshold = 2*self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        mass_difference = np.zeros((n,n),dtype=int)
        # mass_difference = np.round(np.triu(graphnode_moverz[np.newaxis,:] - graphnode_moverz[:,np.newaxis]).flatten())
        for x in range(graphnode_moverz.size - 1):
            if x == 0 and pepmass:
                mass_difference[x, x + 1:] = graphnode_moverz[x + 1:] - pepmass + mass_threshold
            else:
                mass_difference[x, x + 1:] = graphnode_moverz[x + 1:] - graphnode_moverz[x]
        mass_difference = mass_difference.flatten()
        # _, edge_mask_index, _ = np.intersect1d(mass_difference,self.theo_edge_mass,return_indices=True)
        edge_mask_index, _ = np.where(mass_difference[:, None] == self.theo_edge_mass)
        edge_matrix = np.zeros(n**2,dtype=int)
        edge_matrix[edge_mask_index] = mass_difference[edge_mask_index]
        edge_matrix = edge_matrix.reshape(n,n)
        adjacency_matrix = edge_matrix>0
        return edge_matrix, adjacency_matrix.astype(int)
    
def subsequences_starting_and_ending_with_KR(s, enzyme):
    # Identify positions of 'K' and 'R'
    positions = [i for i, c in enumerate(s) if c in enzyme]# {'F', 'W', 'Y', 'E'}]
    
    # Collect subsequences that start and end with K or R
    subsequences = []
    for start in positions:
        for end in positions:
            if start < end:
                subsequences.append(s[start+1:end+1])
    
    return subsequences
def separate_fasta(fasta_file, enzyme, miss_cleavaged):
    motif_n_pep = []
    with open(fasta_file, 'r') as f:
        line = f.readline()
        # print(f.readlines())
        while len(line) != 0:
            if line.startswith('>'):
                name = line
                line = f.readline()
            else:
                proteins =[]
                while (not line.startswith('>')) and len(line) != 0:
                    proteins.append(line.strip())
                    
                    line = f.readline()
                protein_seq = ''.join(proteins)
                split_strings = subsequences_starting_and_ending_with_KR(protein_seq, enzyme)

                # max len < 40; miss_cleavage = 3
                for s in split_strings:
                    if len(s) < 40 and  sum(s.count(i) for i in enzyme) <= miss_cleavaged:
                        pattern = r'[ST]'
                        match = re.search(pattern, s)
                        if match:
                            motif_n_pep.append(s)
    print('number of motif in this sample', len(motif_n_pep))          
    return motif_n_pep
def mask_filter(mass_list, pep_mass=None):
    if pep_mass:
        mask = mass_list >= 0
        return mass_list[mask], mask
def peptide_rescoring(product_ions_moverz, product_ions_intensity, precursor_ion_mass):
    node_pep_y_mass, mask = mask_filter(precursor_ion_mass-Ion.peak2sequencemz(product_ions_moverz,'1y'),precursor_ion_mass)
    node_pep_y_mass_int = product_ions_intensity[mask]
    node_pep_2y_mass, mask = mask_filter(precursor_ion_mass-Ion.peak2sequencemz(product_ions_moverz,'2y'),precursor_ion_mass)
    node_pep_2y_mass_int = product_ions_intensity[mask]

    node_pep_1z_mass, mask = mask_filter(precursor_ion_mass-Ion.peak2sequencemz(product_ions_moverz, '1z'),precursor_ion_mass)
    node_pep_1z_mass_int = product_ions_intensity[mask]
    node_pep_2z_mass, mask = mask_filter(precursor_ion_mass-Ion.peak2sequencemz(product_ions_moverz, '2z'),precursor_ion_mass)
    node_pep_2z_mass_int = product_ions_intensity[mask]
    node_pep_1a_mass, mask = mask_filter(Ion.peak2sequencemz(product_ions_moverz, '1a'),precursor_ion_mass)
    node_pep_1a_mass_int = product_ions_intensity[mask]
    node_pep_1b_mass, mask = mask_filter(Ion.peak2sequencemz(product_ions_moverz, '1b'),precursor_ion_mass)
    node_pep_1b_mass_int = product_ions_intensity[mask]
    
    graphnode_mass = np.concatenate(
            [node_pep_y_mass,node_pep_2y_mass, node_pep_1z_mass, node_pep_2z_mass, node_pep_1a_mass, node_pep_1b_mass])#, node_1z_mass_cterm,node_2z_mass_cterm, node_3z_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
    graphnode_mass_int = np.concatenate([node_pep_y_mass_int,node_pep_2y_mass_int, node_pep_1z_mass_int, node_pep_2z_mass_int, node_pep_1a_mass_int, node_pep_1b_mass_int])#, node_1z_mass_cterm_int, node_2z_mass_cterm_int, node_3z_mass_cterm_int])
    
    unique_values, inverse_indices = np.unique(graphnode_mass, return_inverse=True)
    unique_values =  np.sort(unique_values)
    rel_inten = graphnode_mass_int#/graphnode_mass_int.max()
    # print(graphnode_mass_int.shape,graphnode_mass.shape )
    # Sum the values in array2 grouped by unique values in array1
    # sums = np.bincount(inverse_indices, weights=graphnode_mass_int)
    
    sums = [np.sum(rel_inten[graphnode_mass == value]) for value in unique_values]
    sums = np.array(sums)
    return unique_values, sums

def peptides_filtering(pep, product_ions_moverz, product_ions_intensity, pep_mass):
    pep_theo_mass = np.insert(Residual_seq(pep).step_mass,0,0)
    pep_observe_mass, pep_observe_mass_inten = peptide_rescoring(product_ions_moverz, product_ions_intensity, pep_mass)
    pep_low_index = pep_observe_mass.searchsorted(pep_theo_mass-0.02)
    pep_high_index = pep_observe_mass.searchsorted(pep_theo_mass+0.02)
    # print(low_index, high_index)
    pep_score = 0
    pep_matched = 0
    for i, (pep_low, pep_high) in enumerate(zip(pep_low_index, pep_high_index)):
        
        if pep_low < pep_high:
            # print(theo_mass[i],observe_mass[low : high], low, high)
            pep_matched+= 1
            for j in range(pep_low, pep_high):
                pep_score += np.log(pep_observe_mass_inten[j])*(1-(np.abs(pep_theo_mass[i]-pep_observe_mass[j])/0.05)**4)
    pep_ratio = pep_matched/len(product_ions_moverz)
    pep_score *= pep_matched/len(pep_theo_mass)
    return pep_ratio, pep_score

graph_gen = GraphGenerator(candidate_mass,all_edge_mass)
if __name__=='__main__':
    worker, total_worker, file_path, csv_filename,mode,fasta_file,enzyme,miss_cleavaged = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],int(sys.argv[8])
    
    if os.path.exists(os.path.join(file_path, 'filtered_mgf.pkl')):
        with open(os.path.join(file_path, 'filtered_mgf.pkl'), 'rb') as f:
            all_spectra = pickle.load(f)
            print('mgf file loaded')
            # all_spectra.update(read_mgf(file_path))

        with open(os.path.join(file_path, 'filtered_precursor.pkl'), 'rb') as f:
            all_pre = pickle.load(f)
            print('mgf file loaded')
    else:
        all_spectra, all_pre = read_mgf(file_path)
        print('spectrum read done')
    # '''
    # rescore peptide 
    # motif_n_pep = separate_fasta(fasta_file,enzyme,miss_cleavaged)
    with open('/home/q359zhan/olinked/data/bladder-cancer/peptide.txt', 'r') as f:
        motif_n_pep = f.readlines()
    poss_pep_mass = []
    poss_pep = []
    for pep in motif_n_pep:
        pep = pep.strip().replace('J','N')
        # print(pep)
        pep_mass = Residual_seq(pep).mass + Composition('H2O').mass + Composition('proton').mass
        if pep_mass <=4000:
            poss_pep_mass.append(pep_mass)
            poss_pep.append(pep)
    psm_head = list(all_spectra.keys())
    pep_class = PepMass(poss_pep, poss_pep_mass)
    glycan_freq = dict()
    n_cores = os.cpu_count()
    print('Number of cores:', n_cores)
    with ProcessPoolExecutor(max_workers=n_cores) as exe:
        results= list(exe.map(pep_class.find_match, psm_head))
        print(len(results))
    glycan_freq = dict()
    for i in results:
        for (glycan_mass, pep_mass_given, pep,spec_index) in i:
            glycan_mass = round(glycan_mass, 4)
            if glycan_mass in glycan_freq.keys():
                glycan_freq[glycan_mass].append((spec_index, pep, pep_mass_given))
            else:
                glycan_freq[glycan_mass] = [(spec_index, pep, pep_mass_given)]
    for k in sorted(glycan_freq.keys())[:90]:
        filtered_lst += copy.deepcopy(glycan_freq[k])
    # '''
    # with open('/home/q359zhan/olinked/data/alzheimer/disease-mgf/filtered_lst_all_pep_100.pkl', 'rb') as f:
    #     filtered_lst = pickle.load(f)
    spectra_per_worker = int(len(filtered_lst) / total_worker)
    print('spectra_per_worker', spectra_per_worker)
    start_i, end_i = spectra_per_worker * (worker - 1), spectra_per_worker * worker
    psm_head = filtered_lst[start_i:end_i]
    with open(os.path.join(file_path,f'{csv_filename}_{worker}_all_pep_mass.csv'), 'w') as index_writer:
        if total_worker >=10:
            if worker == 10:
                index_writer.write('Spec Index,Charge,mass,pep mass given,pep,Node Number,MSGP File Name,MSGP Datablock Pointer,MSGP Datablock Length,Glycan mass,Isotope Shift\n')
        else:
            if worker == 1:
                index_writer.write('Spec Index,Charge,mass,pep mass given,pep,Node Number,MSGP File Name,MSGP Datablock Pointer,MSGP Datablock Length,Glycan mass,Isotope Shift\n')
        i=0
        print('number of possible glycans', len(filtered_lst))
        for spec_index, pep, pep_mass_given in psm_head:
            product_ion_info = all_spectra[spec_index]
            precursor_ion_mass = all_pre[spec_index]['precursor_mass']
            precursor_charge = all_pre[spec_index]['precursor_charge']
            product_ions_moverz, product_ions_intensity = copy.copy(product_ion_info['product_ions_moverz']), copy.copy(product_ion_info['product_ions_intensity'])

            for iso in [-1, 0, 1]:#range(-1, 3):
                file_num = i//4000
                if i%4000==0:
                    try: writer.close()
                    except: pass
                    writer = open(os.path.join(file_path,f'{csv_filename}_{worker}_{file_num}_all_pep_mass.msgp'),'wb')
                i += 1
                node_mass, node_input, rel_input, edge_mask = graph_gen(spec_index, product_ions_moverz, product_ions_intensity, precursor_ion_mass+iso*Composition('proton').mass, precursor_charge>2,mode,glycan_data=True, pep_mass=pep_mass_given, decoy=False)
                glycan_mass = precursor_ion_mass+iso*1.0033548378 - pep_mass_given
                
                record = {'node_mass':node_mass,
                        'node_input':node_input,
                        'rel_input':rel_input,
                        'edge_input':edge_mask}
                compressed_data = gzip.compress(pickle.dumps(record))
                # for path in sum_vec:
                #     print(path)
                index_writer.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(spec_index,precursor_charge,precursor_ion_mass+iso*Composition('proton').mass,pep_mass_given,pep, node_mass.size,"{}_{}_{}_all_pep_mass.msgp".format(csv_filename, worker, file_num),writer.tell(),len(compressed_data),glycan_mass,iso))
                writer.write(compressed_data)
                sum_vec = []
