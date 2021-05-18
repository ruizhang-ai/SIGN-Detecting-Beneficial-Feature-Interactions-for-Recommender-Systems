import torch
#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
import numpy as np


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, pred_edges=1, transform=None, pre_transform=None):

        """
        if pred_edges=0, the dataset is used for SIGN/GNN only,
        we store the graph with edges in the .edge file
        """
        self.path = root
        self.dataset = dataset
        self.pred_edges= pred_edges 

        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistical_info = torch.load(self.processed_paths[1])
        self.node_num = self.statistical_info['node_num']
        self.data_num = self.statistical_info['data_num']

    @property
    def raw_file_names(self):
        return ['{}{}/{}.data'.format(self.path, self.dataset, self.dataset), \
                '{}{}/{}.edge'.format(self.path, self.dataset, self.dataset)]

    @property
    def processed_file_names(self):
        if not self.pred_edges:
            return ['{}_edge/{}.dataset'.format(self.dataset, self.dataset), \
                    '{}_edge/{}.info'.format(self.dataset, self.dataset)]
        else:
            return ['{}/{}.dataset'.format(self.dataset, self.dataset), \
                    '{}/{}.info'.format(self.dataset, self.dataset)]


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def read_data(self):
        # handle node and class 
        node_list = []
        label = []
        max_node_index = 0
        data_num = 0
        with open(self.datafile, 'r') as f:
            for line in f:
                data_num += 1
                data = line.split()
                # the first element is the label of the class
                label.append(float(data[0]))
                #the rest of the elements are the nodes
                int_list = [int(data[i]) for i in range(len(data))[1:]]
                node_list.append(int_list)
                if max_node_index < max(int_list):
                    max_node_index = max(int_list)


        if not self.pred_edges:
            edge_list = [[[],[]] for _ in range(data_num)]
            sr_list = []
            # handle edges
            with open(self.edgefile, 'r') as f:
                for line in f:
                    edge_info = line.split()
                    node_index = int(edge_info[0])
                    edge_list[node_index][0].append(int(edge_info[1]))
                    edge_list[node_index][1].append(int(edge_info[2]))
        else:
            edge_list = []
            sr_list = []    #sender_receiver_list, containing node index
            for nodes in node_list:
                edge_l, sr_l = self.construct_full_edge_list(nodes)
                edge_list.append(edge_l)
                sr_list.append(sr_l)

        label = self.construct_one_hot_label(label)

        return node_list, edge_list, label, sr_list, max_node_index + 1, data_num

    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[],[]]         #first for sender, second for receiver
        sender_receiver_list = []
        for i in range(num_node):
            for j in range(num_node)[i:]:
                edge_list[0].append(i)
                edge_list[1].append(j)
                sender_receiver_list.append([nodes[i],nodes[j]])

        return edge_list, sender_receiver_list


    def construct_one_hot_label(self, label):
        """Convert an iterable of indices to one-hot encoded labels."""
        nb_classes = int(max(label)) + 1
        targets = np.array(label, dtype=np.int32).reshape(-1)
        return np.eye(nb_classes)[targets]

    def process(self):
        self.datafile, self.edgefile = self.raw_file_names
        self.node, edge, label, self.sr_list, node_num, data_num = self.read_data()

        data_list = []
        sr_data = []
        for i in range(len(self.node)):
            node_features = torch.LongTensor(self.node[i]).unsqueeze(1)
            x = node_features
            edge_index = torch.LongTensor(edge[i])
            y = torch.FloatTensor(label[i])
            if self.pred_edges:
                sr = torch.LongTensor(self.sr_list[i])     #the sender receiver list, stored in edge_attr
            else:
                sr = []

            data = Data(x=x, edge_index=edge_index, edge_attr=sr,  y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        statistical_info = {'data_num': data_num, 'node_num': node_num}
        torch.save(statistical_info, self.processed_paths[1])


    def node_M(self):
        return self.node_num
    
    def data_N(self):
        return self.data_num

    """
    def len(self):
        return len(self.node)
    def get(self, idx):
        ###
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    """
