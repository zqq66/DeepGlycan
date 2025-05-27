import pickle, copy
import pandas as pd
import sys
import numpy as np
from BasicClass import Residual_seq, Composition, Ion
import more_itertools
import os


def record_filter(mass_list, pep_mass=None):
    if pep_mass:
        mask = mass_list >= 0
        return mass_list[mask], mask
mono_composition = {
    'H': Composition('C6H12O6') - Composition('H2O'),
    'N': Composition('C8H15O6N') - Composition('H2O'),
    'A': Composition('C11H19O9N') - Composition('H2O'),
    'G': Composition('C11H19O10N') - Composition('H2O'),
    'F': (Composition('C6H12O5') - Composition('H2O')),
    'X': (Composition('C5H10O5') - Composition('H2O')),
}
id2mass = {k: v.mass for k, v in mono_composition.items()}
def find_fragments(sequence):
    glyan_ions = {0}
    for i in range(1, len(sequence)+1):
        all_comb = set(more_itertools.distinct_combinations(sequence, i))
        all_comb_mass = set(sum(id2mass[s] for s in ss) for ss in all_comb)
        glyan_ions = glyan_ions.union(all_comb_mass)
    return np.sort(np.array(list(glyan_ions)))


def graphnode_mass_generator(product_ions_moverz, product_ions_intensity, muti_charged, pep_mass=None):

    node_1y_mass_cterm, mask = record_filter(product_ions_moverz, pep_mass)
    node_1y_mass_cterm_int = product_ions_intensity[mask]
    # node_1y_mass_nter _ = record_filter(precursor_ion_mass-node_1y_mass_cterm,precursor_ion_mass)
    # 2y ion
    node_2y_mass_cterm, mask = record_filter(Ion.mass2mz(product_ions_moverz, 2), pep_mass)
    node_2y_mass_cterm_int = product_ions_intensity[mask]

    node_3y_mass_cterm,mask = record_filter(Ion.mass2mz(product_ions_moverz, 3), pep_mass)
    node_3y_mass_cterm_int = product_ions_intensity[mask]
    
    if muti_charged:
        graphnode_mass = np.concatenate(
            [node_1y_mass_cterm,node_2y_mass_cterm, node_3y_mass_cterm])#, node_1z_mass_cterm,node_2z_mass_cterm, node_3z_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
        graphnode_mass_int = np.concatenate([node_1y_mass_cterm_int, node_2y_mass_cterm_int, node_3y_mass_cterm_int])#, node_1z_mass_cterm_int, node_2z_mass_cterm_int, node_3z_mass_cterm_int])
    else:
        graphnode_mass = np.concatenate(
            [node_1y_mass_cterm, node_2y_mass_cterm])#,node_1z_mass_cterm,node_2z_mass_cterm])  # node_1a_mass_nterm,node_1b_mass_nterm,
        graphnode_mass_int = np.concatenate([node_1y_mass_cterm_int, node_2y_mass_cterm_int])#,node_1z_mass_cterm_int, node_2z_mass_cterm_int,])
    # graphnode_mass, counts = np.unique(graphnode_mass-(pep_mass-0.02), return_counts=True)
    unique_values, inverse_indices = np.unique(graphnode_mass, return_inverse=True)
    unique_values =  np.sort(unique_values)
    rel_inten = graphnode_mass_int#/graphnode_mass_int.max()
    # print(graphnode_mass_int.shape,graphnode_mass.shape )
    # Sum the values in array2 grouped by unique values in array1
    # sums = np.bincount(inverse_indices, weights=graphnode_mass_int)
    sums = [np.sum(rel_inten[graphnode_mass == value]) for value in unique_values]
    sums = np.array(sums)
    # intensity_mask = sums>0.01
    # print(sums)
    # unique_values = unique_values[intensity_mask]
   
    return unique_values, sums#[intensity_mask]

def check_signature_ion(glycan):
    signature_ion = []
    if 'G' in glycan:
        signature_ion.append(308.098)
        signature_ion.append(290.087)
        signature_ion.append(511.177)
    
    return np.array(signature_ion)

if __name__=='__main__':

    outputfile, file_path = sys.argv[1],sys.argv[2]
    outputfile_name = outputfile.replace('.csv', '')
    df2 = pd.read_csv(outputfile)
    with open(os.path.join(file_path, 'filtered_mgf.pkl'), 'rb') as f:
        spectrum = pickle.load(f)
    with open(os.path.join(file_path, 'filtered_precursor.pkl'), 'rb') as f:
        precursor = pickle.load(f)
    df2 = df2[df2['mass difference'].astype(float)<=0.02]
    glycan_scores = []
    pep_scores = []
    glycan_ratios = []
    pep_ratios = []
    weighted_scores = []
    df = pd.DataFrame(columns=df2.columns.tolist()+['GlycanScore'])
    for i, row in df2.iterrows():
        # print(row)
        spec = row['Spec']
        
        # precursor_mh = precursor[spec]['precursor_mass']+row['isotope_shift']*Composition('proton').mass
        precursor_mh = float(row['mass'])
        # pep_mass = precursor_mh - float(row['label mass'])
        
        # pep_mass -= Composition('proton').mass
        pep_mass = row['Pep mass']
        pep = row['predict pep']
        theo_mass = pep_mass + find_fragments(row['predict'])
        charge = precursor[spec]['precursor_charge']
        # charge = row['Charge']
        product_ion_info = spectrum[spec]
        product_ions_moverz, product_ions_intensity = copy.copy(product_ion_info['product_ions_moverz']), copy.copy(product_ion_info['product_ions_intensity'])#/product_ion_info['product_ions_intensity'].max()
        product_ions_moverz_array = np.array(product_ions_moverz)
        signature_ion = check_signature_ion(row['predict'])
        start_sig = np.searchsorted(product_ions_moverz_array, signature_ion-0.02)
        end_sig = np.searchsorted(product_ions_moverz_array, signature_ion+0.02)
        
        if not np.any(end_sig - start_sig > 0):
            
            # glycan_scores.append(glycan_score)
            continue
        print(spec,'start_sig',product_ions_moverz_array[start_sig], product_ions_intensity[start_sig])
        # peptide rescoring
        # pep = row['Peptide'].replace('J', 'N')
        pep_theo_mass = np.insert(Residual_seq(pep).step_mass,0,0)
        # np.set_printoptions(threshold=1e5)
        # print(observe_mass, pep_theo_mass)
        
        observe_mass, observe_mass_inten = graphnode_mass_generator(product_ions_moverz, product_ions_intensity, int(charge),pep_mass)
        low_index = observe_mass.searchsorted(theo_mass-0.02)
        high_index = observe_mass.searchsorted(theo_mass+0.02)
        # print(low_index, high_index)
        glycan_score = 0
        matched = 0
        for i, (low, high) in enumerate(zip(low_index, high_index)):
            
            if low < high:
                # print(theo_mass[i],observe_mass[low : high], low, high)
                matched+= 1
                for j in range(low, high):
                    glycan_score += np.log(observe_mass_inten[j])*(1-(np.abs(theo_mass[i]-observe_mass[j])/0.05)**4)
        glycan_score *= matched/len(theo_mass)
        print('glycan_score', glycan_score)
        df.loc[len(df)] = row.tolist()+[glycan_score]
        glycan_scores.append(glycan_score)
    # break
    # df2.insert(len(df2.columns), 'glycan rescoring',glycan_scores,  True)

    max_idx = df.groupby('Spec')['GlycanScore'].idxmax()
    result_df = df.loc[max_idx]
    result_df.to_csv(outputfile+'rescoring_max.csv',index=False)