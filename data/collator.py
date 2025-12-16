import torch
from torch.nn.functional import pad


class DGCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        nodenum = [record['node_num'] for record in batch]
        max_nodenum = max(nodenum)

        node_feature = torch.stack(
            [pad(record['node_feature'], (0, 0, 0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        node_sourceion = torch.stack(
            [pad(record['node_sourceion'], (0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        node_mass = torch.stack(
            [pad(record['node_mass'], (0, max_nodenum - nodenum[i])) for i, record in enumerate(batch)])
        node_mass[:, 0] = 0
    

        encoder_input = {
            'node_feature': node_feature,
            'node_sourceion': node_sourceion,
            'node_mass': node_mass,
            # 'rel_mask': rel_mask
        }

        tgt = torch.stack([torch.tensor(record['tgt']) for record in batch])
        glycan_mass_embeded = torch.stack([record['glycan_mass_embeded'] for record in batch])
        glycan_crossattn_mass = torch.stack([record['glycan_crossattn_mass'] for record in batch])
        pos_index = torch.arange(0, 1).unsqueeze(0)

        precursor_mass = [record['precursor_mass'] for record in batch]
        glycan_mass =  [record['glycan_mass'] for record in batch]
        pep_mass = [record['pep_mass'] for record in batch]
        charge = [record['precursor_charge'] for record in batch]
        pep_seq = [record['pep_seq'] for record in batch]
        psm_index = [record['psm_index'] for record in batch]
        isotope_shift = [record['isotope_shift'] for record in batch]
        decoder_input = {'tgt': tgt,
                         'pos_index': pos_index,
                         'glycan_crossattn_mass': glycan_crossattn_mass,
                         'glycan_mass': glycan_mass_embeded,
                         'node_mass': node_mass,
                         }

        return encoder_input, decoder_input,pep_seq, precursor_mass, pep_mass,glycan_mass, psm_index,isotope_shift, charge
