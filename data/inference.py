import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from .BasicClass import Composition
import operator
from copy import deepcopy
import math
import numpy as np

@dataclass
class Pep_Inference_Status:
    psm_idx: str
    idx: int
    inference_seq: list[int]
    #label_seq: list[str]
    mass_list: list[float]
    nmass:torch.Tensor
    precursor_mass: float
    ms1_threshold: float
    ms2_threshold: float
    precursor_mass: float
    pep_mass: float
    glycan_mass:float
    pep: str
    isotope_shift: int
    score_list: list = field(default_factory=list)
    current_mass: float = 0
    score: float = 0
    total_score: float = 0
    total_inference_len: float = 0

@dataclass
class Pep_Finish_Status:
    psm_idx: str
    inference_seq: str
    #label_seq: str
    isotope_shift: int
    score_list: list
    score: float
    mass_diff:float
    precursor_mass:float
    pep:str
    pep_mass:float
    report_mass:float

class Pep_Inference_BeamPool(object):
    def __init__(self, max_size):
        self.pool = []
        self.max_size = max_size

    def is_empty(self): return len(self.pool) == 0

    def put(self, data):
        self.pool.append(data)

    def get(self):
        return self.pool.pop(0)

    def sort(self):
        self.pool = sorted(self.pool,
                           key=operator.attrgetter('score'),
                           reverse=True)[:self.max_size]

class OptimisedInference(object):
    def __init__(self, cfg, model, inference_dl, aa_dict, tokenize_aa_dict, detokenize_aa_dict, mass_candidates):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.aa_dict = aa_dict
        self.tokenize_aa_dict = tokenize_aa_dict
        self.detokenize_aa_dict = detokenize_aa_dict
        self.prediction_rate = []
        self.unable2predict = set()
        self.range_tensor = torch.arange(self.cfg.data.peptide_max_len)[1::2]
        self.range_tensor = self.range_tensor.repeat_interleave(2)
        self.range_tensor = torch.concat((torch.tensor([0]), self.range_tensor))
        self.beam_size=self.cfg.inference.beam_size
        self.isotope_check = self.cfg.inference.isotope
        self.mass_cand = [int(i) for i in self.cfg.inference.mass_cand.split(',')]

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        pep_finish_pool = {}
        tgt, mems_ori, pep_status_list, device, psm_index = self.exploration_initializing()

        pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
        if len(pep_status_list) == 0:
            return pep_finish_pool
        tgt, glycan_mass, glycan_crossattn_mass,parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, device)
        mem = self.past_input_stacker(pep_status_list, mems_ori)

        while len(pep_status_list)>0:

            with torch.inference_mode():
                if dist.is_initialized():
                    tgt = self.model.module.tgt_get(tgt=tgt,
                                         src=mem,
                                         pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                         glycan_crossattn_mass=glycan_crossattn_mass,
                                         glycan_mass=glycan_mass,
                                         parent_mono_lists=parent_mono_lists)
                else:
                    tgt = self.model.tgt_get(tgt=tgt,
                                         src=mem,
                                         pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                         glycan_crossattn_mass=glycan_crossattn_mass,
                                         glycan_mass=glycan_mass,
                                         parent_mono_lists=parent_mono_lists)
            pep_status_list, pep_finish_pool = self.next_aa_choice(tgt, pep_status_list, pep_finish_pool)
            if len(pep_status_list)<=0: break
            tgt, glycan_mass, glycan_crossattn_mass,parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, device)
            mem = self.past_input_stacker(pep_status_list, mems_ori)
        # print('pep_finish_pool', pep_finish_pool)
        return pep_finish_pool

    def exploration_initializing(self):
        
        encoder_input, _, pep, precursor_mass, pep_mass,glycan_mass, psm_index,isotope_shift,charge_threshold  = next(self.inference_dl)
        
        with torch.inference_mode(): 
            if dist.is_initialized():
                mem = self.model.module.mem_get(**encoder_input)
            else:
                mem = self.model.mem_get(**encoder_input)
        pep_status_list = []
        for i in range(len(psm_index)):

            # print('mass_block', mass_block, seq[i])
            pool = Pep_Inference_BeamPool(max_size=self.beam_size)
            pep_status_list.append(pool)

            if self.cfg.inference.ms1_threshold_unit=='ppm':
                ms1_threshold = (precursor_mass[i]+Composition('H2O').mass+\
                                 charge_threshold[i]*Composition('proton').mass) * self.cfg.inference.ms1_threshold * 1e-6

            elif self.cfg.inference.ms1_threshold_unit=='Th': ms1_threshold = self.cfg.inference.ms1_threshold
            else: raise NotImplementedError

            if self.cfg.inference.ms2_threshold_unit=='Th': ms2_threshold = self.cfg.inference.ms2_threshold*2
            else: raise NotImplementedError

            pool.put(Pep_Inference_Status(psm_idx=psm_index[i],idx=i,
                                          glycan_mass=glycan_mass[i],
                                          precursor_mass=precursor_mass[i],
                                          ms1_threshold = ms1_threshold,
                                          ms2_threshold = ms2_threshold,
                                          inference_seq=[5], mass_list=[0.],
                                          nmass=torch.tensor([0.]), current_mass=0,
                                          pep=pep[i],
                                          pep_mass=pep_mass[i],
                                          isotope_shift=isotope_shift[i]))
        tgt, glycan_mass, glycan_crossattn_mass, parent_mono_lists = self.decoder_inference_input_gen(pep_status_list, mem.device)
        with torch.inference_mode():
            if dist.is_initialized():
                tgt = self.model.module.tgt_get(tgt=tgt,
                                         src=mem,
                                        pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                     glycan_mass=glycan_mass,
                                     glycan_crossattn_mass=glycan_crossattn_mass,
                                        parent_mono_lists=parent_mono_lists)
            else:
                tgt = self.model.tgt_get(tgt=tgt,
                                     src=mem,
                                     pos_index=self.range_tensor[:tgt.shape[1]].cuda().unsqueeze(0),
                                     glycan_mass=glycan_mass,
                                     glycan_crossattn_mass=glycan_crossattn_mass,
                                        parent_mono_lists=parent_mono_lists)

        return tgt, mem, pep_status_list, mem.device, psm_index
    
    def next_aa_choice(self, tgt, pep_status_list, pep_finish_pool):
        i = 0
        new_pep_status_list = []
        tgt = tgt.float()
        for current_status_pool in pep_status_list:
            pool = Pep_Inference_BeamPool(max_size=current_status_pool.max_size)
            for current_status in current_status_pool.pool:
                next_aa = tgt[i, -1]

                for aa_id in self.mass_cand:
                    current_status_new = deepcopy(current_status)
                    current_status_new.inference_seq.append(aa_id)
                    current_mass = self.detokenize_aa_dict[aa_id]
                    current_status_new.mass_list.append(current_mass)
                    mass_list = torch.tensor(current_status_new.mass_list)
                    current_status_new.current_mass += current_mass
                    current_status_new.total_score += next_aa[aa_id]
                    current_status_new.total_inference_len+=1
                    current_status_new.score = current_status_new.total_score/len(current_status.inference_seq)
                    current_status_new.score_list += [next_aa[aa_id]]
                    current_status_new.nmass=torch.cumsum(mass_list, dim=0)
                    #print('mass-diff', current_status_new.precursor_mass-current_status_new.current_mass)
                    if (abs(current_status_new.glycan_mass-current_status_new.current_mass)< 1 or \
                        current_status_new.glycan_mass<current_status_new.current_mass )or \
                        len(current_status_new.inference_seq)>=self.cfg.data.peptide_max_len:
                        # print('mass-diff', current_status_new.precursor_mass-current_status_new.current_mass)
                        mass_diff = abs(current_status_new.glycan_mass-current_status_new.current_mass)
                        # print('tgt',current_status_new.psm_idx, current_status_new.inference_seq, current_status_new.score, current_status_new.current_mass, current_status_new.glycan_mass)
                        # print('mass_diff', mass_diff)
                        if current_status_new.idx in pep_finish_pool:
                            if mass_diff<pep_finish_pool[current_status_new.idx].mass_diff:# or pep_finish_pool[current_status_new.idx].score<current_status_new.score:
                                # pep_finish_pool[current_status_new.idx].mass_diff
                                # print(current_status_new.idx, current_status_new.score_list, mass_diff, current_status_new.inference_seq[1:])
                                pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(psm_idx=deepcopy(current_status_new.psm_idx),
                                                                                        inference_seq=deepcopy(current_status_new.inference_seq[1:]),
                                                                                        precursor_mass = current_status_new.precursor_mass,
                                                                                        report_mass = current_status_new.current_mass,
                                                                                        pep=current_status_new.pep,
                                                                                        pep_mass=current_status_new.pep_mass,
                                                                                        #label_seq=deepcopy(current_status_new.label_seq),
                                                                                        score=current_status_new.score,
                                                                                        score_list=deepcopy(current_status_new.score_list),
                                                                                        mass_diff=mass_diff,
                                                                                        isotope_shift=current_status_new.isotope_shift)
                        else:
                            pep_finish_pool[current_status_new.idx] = Pep_Finish_Status(psm_idx=deepcopy(current_status_new.psm_idx),
                                                                                        inference_seq=deepcopy(current_status_new.inference_seq[1:]),
                                                                                        #label_seq=deepcopy(current_status_new.label_seq),
                                                                                        score=current_status_new.score,
                                                                                         precursor_mass = current_status_new.precursor_mass,
                                                                                            report_mass = current_status_new.current_mass,
                                                                                             pep=current_status_new.pep,
                                                                                            pep_mass=current_status_new.pep_mass,
                                                                                        score_list=deepcopy(current_status_new.score_list),
                                                                                        mass_diff=mass_diff,  
                                                                                        isotope_shift=current_status_new.isotope_shift)
                    else:
                        # print(current_status_new)
                        pool.put(current_status_new)

                i+=1
            if len(pool.pool)>0:
                pool.sort()
            
                new_pep_status_list.append(pool)
        
        return new_pep_status_list, pep_finish_pool


    def decoder_inference_input_gen(self, pep_status_list, device):
        tgt = []
        current_mass = []
        current_cross_atte_mass = []
        parent_mono_lists = []
        for current_status_pool in pep_status_list:
            for current_status in current_status_pool.pool:
                tgt.append(torch.tensor(current_status.inference_seq))
                mass_list = torch.tensor(current_status.mass_list)
                cterm_mass = current_status.glycan_mass - current_status.nmass
                current_mass.append(torch.stack([current_status.nmass,cterm_mass],dim=1))
                current_cross_atte_mass.append(torch.stack([current_status.nmass,mass_list],dim=1))
                parent_mono_lists.append(mass_list)

        tgt = torch.stack(tgt)
        pep_mass = torch.stack(current_mass)
        pep_crossattn_mass = torch.stack(current_cross_atte_mass)
        parent_mono_list = torch.stack(parent_mono_lists).to(device)

        return tgt.cuda(), pep_mass.cuda(), pep_crossattn_mass.cuda(), parent_mono_list

    def tonkenize(self, inference_seq):
        tgt = torch.LongTensor([self.aa_dict['<bos>']]+[self.aa_dict[aa] for aa in inference_seq[:-1]])
        return tgt

    def past_input_stacker(self, pep_status_list, mems_ori):
        idx_list = []
        for current_status_pool in pep_status_list:
            for current_status in current_status_pool.pool:
                idx_list.append(current_status.idx)
        idx_list = torch.tensor(idx_list,dtype=torch.long,device=mems_ori.device)
        mem = mems_ori.index_select(0,idx_list)
        return mem