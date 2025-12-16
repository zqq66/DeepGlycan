import os
import gzip
import torch
import pickle
from torch.utils.data import Dataset


class DGDataset(Dataset):
    def __init__(self, cfg, aa_dict, all_psms):
        super().__init__()
        self.cfg = cfg
        self.aa_dict = aa_dict
        self.all_psms = all_psms

    def __getitem__(self, idx):
    
        spec = self.all_psms[idx]
        # print(type(spec['node_mass']))
        node_mass = torch.Tensor(spec['node_mass'])
        # node_mass = torch.round(node_mass, decimals=3)

        precursor_mass = spec['precursor_mass']
        pep_mass = spec['pep_mass']
        charge = spec['charge']
        pep = spec['pep']

        node_feature = torch.as_tensor(spec['node_input']['node_feat'], dtype=torch.bfloat16)
        node_sourceion = torch.as_tensor(spec['node_input']['node_sourceion'],dtype=torch.long)
        # print(node_feature.shape, node_sourceion.shape, node_mass.shape)

        # decoder input
        glycan_mass = precursor_mass - pep_mass #precursor_mass
        isotope = spec['isotope_shift']
        tgt = [0] 
        glycan_mass_embeded = torch.DoubleTensor([0, glycan_mass])
        glycan_crossattn_mass = torch.concat(
            [torch.DoubleTensor([0, 0]) / i for i in range(1, self.cfg.model.max_charge + 1)], dim=-1)

        return {'node_num': spec['node_mass'].size,
                'node_feature': node_feature,
                'node_sourceion': node_sourceion,
                'node_mass': node_mass,
                'glycan_mass_embeded': glycan_mass_embeded,
                'glycan_crossattn_mass': glycan_crossattn_mass,
                'glycan_mass': glycan_mass,
                'pep_seq': pep,
                'precursor_mass': precursor_mass,#+ori_idx % self.shift_range+self.cfg.inference.min_isotope_shift,
                'isotope_shift': isotope,
                'pep_mass': pep_mass,
                'precursor_charge': charge,
                'psm_index': spec['scan'],
                'tgt': tgt,
                }

    def __len__(self):
        return len(self.all_psms)
