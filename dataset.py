import os
import gzip
import torch
import pickle
from torch.utils.data import Dataset


class DGDataset(Dataset):
    def __init__(self, cfg, aa_dict, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        self.aa_dict = aa_dict
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.loc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            try:
                spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))
            except Exception as error:
                print('spec_header', spec_head)
                print(error)

        node_mass = torch.Tensor(spec['node_mass'])
        glycan_mass = spec_head['Glycan mass']
        precursor_mass = spec_head['mass']
        pep_mass = spec_head['pep mass given']
        charge = spec_head['Charge']
        pep = spec_head['pep']

        node_feature = spec['node_input']['node_feat'][:, 0, :-1]
        node_sourceion = spec['node_input']['node_sourceion'][:, 0]
        node_sourceion[node_sourceion == 10] = 4
        node_sourceion[node_sourceion == 13] = 5

        dist = spec['rel_input']['dist'].long()
        predecessors = spec['rel_input']['predecessors'].long()
        predecessors[predecessors == -9999] = 0

        # decoder input
        if 'graph_label' in spec.keys():
            seq = spec_head['Glycan Stru']
            graph_label = torch.tensor(spec['graph_label'])
            isotope = None
        else:
            graph_label = None
            seq = None
            isotope = spec_head['Isotope Shift']

        tgt = [0] 
        glycan_mass_embeded = torch.DoubleTensor([0, glycan_mass])
        glycan_crossattn_mass = torch.concat(
            [torch.DoubleTensor([0, 0]) / i for i in range(1, self.cfg.model.max_charge + 1)], dim=-1)

        return {'node_num': spec_head['Node Number'],
                'node_feature': node_feature,
                'node_sourceion': node_sourceion,
                'node_mass': node_mass,
                'dist': dist,
                'predecessors': predecessors,
                'graph_label': graph_label,
                'tgt': tgt,
                'seq': seq,
                'glycan_mass_embeded': glycan_mass_embeded,
                'glycan_crossattn_mass': glycan_crossattn_mass,
                'glycan_mass': glycan_mass,
                'precursor_mass': precursor_mass,#+ori_idx % self.shift_range+self.cfg.inference.min_isotope_shift,
                'isotope_shift': isotope,
                'pep_mass': pep_mass,
                'precursor_charge': charge,
                'psm_index': spec_head['Spec Index'],
                'pep': pep
               # 'rank': spec_head['glycan']
                }

    def __len__(self):
        return len(self.spec_header) * self.shift_range
