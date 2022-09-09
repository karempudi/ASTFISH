# tracking nets, new graph nets..
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

class trackerNet(nn.Module):
    
    def __init__(self, input_node_size= 6, hidden_size=64, edge_classes=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_classes = edge_classes
        self.input_node_size = input_node_size
        self.edge_mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_size, edge_classes),
        )
        
        self.inital_node_transform = nn.Sequential(
                    #nn.BatchNorm1d(input_node_size),
                    nn.Linear(input_node_size, hidden_size),
                    nn.ReLU(inplace=True),
        )
            
    def forward(self, one_step_data):
        x, edge_index, edge_attr = one_step_data['x'], one_step_data['edge_index'], one_step_data['edge_attr']
        n_objects_1, n_objects_2 = one_step_data['node_t'], one_step_data['node_t1']
        
        # initial node transform
        x = self.inital_node_transform(x)
        
        # update the edges accordingly and generate the affinity matrices
        src, dst = edge_index
        #print("src:", src, "dst:", dst)
        #print("x_src shape:", x[src].shape)
        #print("x_dst shape:", x[dst].shape)
        diff = x[src] - x[dst]
        #print("Diff shape:", diff.shape)
        edge_attr_mlp = self.edge_mlp(diff)
        affinity_matrix_scores = edge_attr_mlp.view(n_objects_1, n_objects_2, self.edge_classes)
        #print("Affinity matrix size: ", affinity_matrix_scores.shape)
                
        return x, affinity_matrix_scores #(n_objects_1, n_objects_2), edge_index

model_dict = {
    'tracker': trackerNet
}