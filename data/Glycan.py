import copy
import json


mono_names = ['hex', 'hexNAc', 'neuAc', 'neuGc', 'fuc']


def convert2glycoCT(structure_encoding):
    idx = 0
    # for s in structure_encoding:
    #     if s.islower():
    #         structure_encoding = structure_encoding.replace(s, ']')
    #     elif s.isupper():
    #         structure_encoding = structure_encoding.replace(s, '[')
    structure_encoding = structure_encoding.replace(')', ']')
    structure_encoding = structure_encoding.replace('(', '[')
    p_lst= ['H','N','A','G','F']
    for i in p_lst:
        if i in structure_encoding:
            structure_encoding = structure_encoding.replace(i,str(p_lst.index(i)))
    for s in structure_encoding[1:]:
        if s == '[':
            temp_lst = list(structure_encoding)
            temp_lst.insert(idx + 1, ',')
            idx += 2
            structure_encoding = "".join(temp_lst)
        else:
            idx += 1
    struc_lst = json.loads(structure_encoding)
    # print(mono_names[struc_lst[0] - 1])
    bos = Monosaccharide('<bos>')
    root = Monosaccharide(mono_names[struc_lst[0]])
    root = construct_glycan(root, struc_lst[1:])
    bos.add_child(root)
    return bos

def count_letters(input_string):
    letter_count = 0
    for char in input_string:
        if char.isalpha():
            letter_count += 1
    return letter_count

def construct_glycan(root, struc_lst):
    for i, s in enumerate(struc_lst):
        mono = Monosaccharide(mono_names[s[0]])
        root.add_child(mono)
        construct_glycan(mono, s[1:])
    return root


class Monosaccharide:
    def __init__(self,name, index=None,parent=None, children=None):
        self.parent = parent
        self.children = children
        self.name = name
        self.num_children = 0
        self.index = index

        # self.mass = Residual_seq.__aa_residual_composition[self.name].mass

    def __len__(self):
        self.num_children = self.count_nodes()
        return self.num_children

    def count_nodes(self):
        if self.children is None:
            return 1
        else:
            # Count the current node and recursively count nodes in left and right subtrees
            return 1 + sum([r.count_nodes() for r in self.children])

    def add_child(self, child_mono):
        if self.children is None:
            self.children = [child_mono]
        else:
            self.children.append(child_mono)
        child_mono.parent = self

    def find_children(self):
        return copy.deepcopy(self.children)

    def set_index(self, idx):
        self.index = idx
    def find_children_by_idx(self, index):
        current_nodes = [self]
        while len(current_nodes) != 0:
            current_node = current_nodes.pop(0)
            if current_node.index == index:
                return current_node.children
            elif current_node.index > index:
                current_nodes += [current_node.parent] if current_node.parent is not None else []
            elif current_node.index < index:
                current_nodes += current_node.children if current_node.children is not None else []
            current_nodes = [i for i in current_nodes if i.index is not None]
        return None
    def find_first_unassigned_node(self):
        current_nodes = [self]
        while len(current_nodes) != 0:
            current_node = current_nodes.pop(0)
            if current_node.index is None:
                return current_node
            else:
                current_nodes += current_node.children if current_node.children is not None else []
        return None

    def find_first_unassigned_mono_with_mf(self, mono):
        current_nodes = [self]
        while len(current_nodes) != 0:
            current_node = current_nodes.pop(0)
            if current_node.index is None and current_node.name == mono:
                return current_node
            else:
                current_nodes += current_node.children if current_node.children is not None else []
        return None

    def find_first_unassigned_mono(self, mono):
        current_nodes = [self]
        while len(current_nodes) != 0:
            current_node = current_nodes.pop(0)
            if current_node.parent is not None and current_node.parent.index is None:
                return
            if current_node.index is None and current_node.name == mono:
                return current_node
            else:
                current_nodes += current_node.children if current_node.children is not None else []
        return None
