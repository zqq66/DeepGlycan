import os
import sys
import re
import gzip
import json
import glypy
import torch
import pickle
import numpy as np
import pandas as pd
import more_itertools
from glob import glob
import itertools as it
from sys import getsizeof
from BasicClass import Residual_seq, Ion, Composition
from itertools import combinations_with_replacement
from edge_matrix_gen import typological_sort_floyd_warshall
from glypy.structure.glycan import fragment_to_substructure
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
        self.c_term_ion_list = ['1y','1y-H2O', '1y-H2O','2y','2y-H2O','2y-H2O', '3y', '3y-H2O']
        self.label_ratio = []
        
    def __call__(self, scan, glycan, product_ions_moverz, product_ions_intensity, precursor_ion_mass, muti_charged, glycan_data=False, pep_mass=None):
        peak_feature = self.peak_feature_generation(product_ions_moverz, product_ions_intensity)
        subnode_mass, subnode_feature = self.candidate_subgraph_generator(precursor_ion_mass, product_ions_moverz, peak_feature)
        if glycan_data:

            node_mass = self.graphnode_mass_generator(precursor_ion_mass, product_ions_moverz, muti_charged, pep_mass)
            # subedge_maxnum, edge_type, edge_error = self.edge_generator(node_mass, precursor_ion_mass, pep_mass)
            edge_matrix, edge_mask = self.edge_generator(node_mass, pep_mass)
        else:
            node_mass = self.graphnode_mass_generator(precursor_ion_mass, product_ions_moverz, muti_charged)
            # subedge_maxnum, edge_type, edge_error = self.edge_generator(node_mass, precursor_ion_mass)
            edge_matrix, edge_mask = self.edge_generator(node_mass)

        #assert node_mass.size<=512
        # all neighbourhood edge are counted as node feature
        node_feat, node_sourceion = self.graphnode_feature_generator(node_mass, subnode_mass, subnode_feature, precursor_ion_mass)

        if glycan_data:
            graph_labels = self.graph_label_generator(glycan, node_mass, precursor_ion_mass)

        else:
            graph_labels = self.graph_label_generator(glycan, node_mass, precursor_ion_mass)
            graph_labels = torch.IntTensor(graph_labels)
        dist, predecessors = typological_sort_floyd_warshall(edge_mask)
        if np.isnan(node_feat).any() or np.isnan(node_sourceion).any():
            print(scan)
        node_input = {'node_feat': torch.Tensor(node_feat),
                      'node_sourceion': torch.IntTensor(node_sourceion)}
        edge_input = {'edge_feat': torch.Tensor(edge_matrix),
                      'adj_matrix': torch.IntTensor(edge_mask)}
        rel_input = {'dist': torch.Tensor(dist),
                     'predecessors': torch.Tensor(predecessors),}
        return node_mass, node_input, rel_input, edge_input, graph_labels

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

    def graphnode_mass_generator(self, precursor_ion_mass, product_ions_moverz, muti_charged, pep_mass=None):
        # our model is trained on ethcd data
        node_1y_mass_cterm, _ = self.record_filter(product_ions_moverz, precursor_ion_mass, pep_mass)
        # node_1y_mass_nterm, _ = self.record_filter(precursor_ion_mass-node_1y_mass_cterm,precursor_ion_mass)
        # 2y ion
        node_2y_mass_cterm, _ = self.record_filter(Ion.mass2mz(product_ions_moverz, 2), precursor_ion_mass, pep_mass)
        node_3y_mass_cterm, _ = self.record_filter(
            Ion.mass2mz(product_ions_moverz, 3), precursor_ion_mass, pep_mass)
        # node_1z_mass_cterm, _ = self.record_filter(product_ions_moverz+Composition('O').mass,precursor_ion_mass, pep_mass)
        # node_2z_mass_cterm, _ = self.record_filter(node_2y_mass_cterm+ Composition('O').mass, precursor_ion_mass,pep_mass)
        # node_3z_mass_cterm, _ = self.record_filter(node_3y_mass_cterm+ Composition('O').mass,precursor_ion_mass, pep_mass)

        if muti_charged:
            graphnode_mass = np.concatenate(
                [node_1y_mass_cterm,node_2y_mass_cterm, node_3y_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
        else:
            graphnode_mass = np.concatenate(
                [node_1y_mass_cterm, node_2y_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
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

    
    def graph_label_generator(self, glycans, node_mass, precursor_ion_mass):
        theo_glopeptide_mass = [0]
        # print('glycans',glycans)
        glyan_ions = [0]
        for ion in glycans.fragments(kind='Y', max_cleavages=5):
            # print(ion.kind, ion.mass, [MonosaccharideResidue.from_monosaccharide(node).residue_name() for node in fragment_to_substructure(ion, glycan).iternodes(method='bfs')])
            glyan_ions.append(ion.mass - Composition('H2O').mass)
        theo_glopeptide_mass = list(map(sum, it.product(glyan_ions, theo_glopeptide_mass)))
            # theo_glopeptide_mass = np.add.outer(theo_glopeptide_mass, glyan_ions).flatten()
        theo_glopeptide_mass = np.unique(theo_glopeptide_mass)
        # theo_node_mass = np.concatenate((theo_pep_node_mass, theo_intact_glycan_mass, theo_glopeptide_mass))
        mass_threshold = self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        start_index = node_mass.searchsorted(theo_glopeptide_mass-mass_threshold)
        end_index = node_mass.searchsorted(theo_glopeptide_mass+mass_threshold)
        graph_label = np.zeros((node_mass.size,theo_glopeptide_mass.size))
        label_num = 0
        for i, (lower_bound, higher_bound) in enumerate(zip(start_index, end_index)):
            graph_label[:,i][lower_bound:higher_bound] = 1
            if higher_bound > lower_bound:
                label_num += 1
            # print(lower_bound, node_mass[lower_bound], higher_bound, node_mass[higher_bound])
        # print(node_mass[np.any(graph_label != 0, axis=-1)])
        graph_label[0] = 1
        graph_label[-1] = 1
        return graph_label

mono_names = ['Man', 'GlcNAc', 'NeuAc', 'NeuGc', 'Fuc', 'Xyl']
def convert2glycoCT(structure_encoding):
    idx = 0
    
    structure_encoding = structure_encoding.replace(')', ']')
    structure_encoding = structure_encoding.replace('(', '[')
    p_lst= ['H','N','A','G','F', 'X']
    for i in p_lst:
        if i in structure_encoding:
            structure_encoding = structure_encoding.replace(i,str(p_lst.index(i) + 1))
    for s in structure_encoding[1:]:
        if s == '[':
            temp_lst = list(structure_encoding)
            temp_lst.insert(idx + 1, ',')
            idx += 2
            structure_encoding = "".join(temp_lst)
        else:
            idx += 1
    try:
        struc_lst = json.loads(structure_encoding)
    except:
        print(structure_encoding)
        return False
        
    root = mono_names[struc_lst[0] - 1]
    glycan = glypy.Glycan(root=glypy.monosaccharides[root])
    glycan = construct_glycan(glycan.root, glycan, struc_lst[1:], 0)
    return glycan

def construct_glycan(root, glycan, struc_lst, cur_idx):

    for i, s in enumerate(struc_lst):
        mono = glypy.monosaccharides[mono_names[s[0]-1]]
        root.add_monosaccharide(mono)
        next_idx = cur_idx+ 1
        glycan.reindex(method='dfs')
        root2 = glycan[next_idx]
        construct_glycan(root2, glycan, s[1:], next_idx)
    return glycan


graph_gen = GraphGenerator(candidate_mass,all_edge_mass)

if __name__=='__main__':
    worker, total_worker, file_path, csv_filename = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    psm_head = pd.read_csv(os.path.join(file_path, csv_filename), index_col='Scan', delimiter='\t')
    print(len(psm_head), getsizeof(psm_head))
    spectra_per_worker = int(len(psm_head)/total_worker)
    print('spectra_per_worker', spectra_per_worker)
    start_i, end_i = spectra_per_worker*(worker-1), spectra_per_worker*worker
    psm_head = psm_head.iloc[start_i:end_i]
    print(len(psm_head), getsizeof(psm_head))

    if os.path.exists(os.path.join(file_path, 'all_mgf.pkl')):
        with open(os.path.join(file_path, 'all_mgf.pkl'), 'rb') as f:
            all_spectra = pickle.load(f)
            print('mgf file loaded')
    else:
        all_spectra = read_mgf(file_path)
        print('spectrum read done')
    print(list(all_spectra.keys())[0])
    with open(os.path.join(file_path,f'{csv_filename}_{worker}_train.csv'), 'w') as index_writer:
        if worker == 1:
            index_writer.write('Spec Index,Annotated Sequence,Charge,mass,Node Number,MSGP File Name,MSGP Datablock Pointer,MSGP Datablock Length,Glycan Stru,Pep mass,Glycan mass\n')
        for i, (spec_index, (seq, precursor_charge, precursor_ion_mass, glycan_ids, glycan_sites, source_file, glycan_mass, pep_mass_given)) in enumerate(
                psm_head[['Peptide', 'Charge', 'PrecursorMH', 'PlausibleStruct', 'GlySite', 'RawName', 'GlyMass', 'PeptideMH']].iterrows()):

            file_num = i//4000
            if i%4000==0:
                try: writer.close()
                except: pass 
                writer = open(os.path.join(file_path,f'{csv_filename}_{worker}_{file_num}_train.msgp'),'wb')
            if 'p' in glycan_ids:
                continue
            seq = seq.replace('L','I').replace(' ','')
            seq = seq.replace('J', 'N')
            seq = re.sub('[^a-zA-Z]+', '', seq)
            spec_index = source_file +'.raw'+ str(int(spec_index))
            product_ion_info = all_spectra[spec_index]
            # if glycan_mass + pep_mass_given - precursor_ion_mass > 1:
            #     print(spec_index, glycan_mass + pep_mass_given - precursor_ion_mass)
            product_ions_moverz, product_ions_intensity = product_ion_info['product_ions_moverz'], product_ion_info['product_ions_intensity']
            # glycans = list(re.sub('[^a-zA-Z]+', '', glycan_ids))
            glycans = convert2glycoCT(glycan_ids)
            node_mass, node_input, rel_input, edge_mask, graph_labels = graph_gen(spec_index, glycans, product_ions_moverz, product_ions_intensity, precursor_ion_mass, precursor_charge>2,glycan_data=True, pep_mass=pep_mass_given)
            record = {'node_mass':node_mass,
                      'node_input':node_input,
                      'rel_input':rel_input,
                      'edge_input':edge_mask,
                      'graph_label':graph_labels.any(axis=1)}
            compressed_data = gzip.compress(pickle.dumps(record))

            index_writer.write('{},{},{},{},{},{},{},{},{},{},{} \n'.format(spec_index,seq,precursor_charge,precursor_ion_mass,node_mass.size,"{}_{}_{}_train.msgp".format(csv_filename, worker, file_num),writer.tell(),len(compressed_data),glycan_ids, pep_mass_given,glycan_mass))
            writer.write(compressed_data)
            sum_vec = []
