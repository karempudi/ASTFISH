import torch
import torch.nn as nn
import numpy as np

def pairwise_distance(node_features, objects):
    """
    Arguments:
        node_features: (N x feature_size), where N is number of nodes
            and feature_size is the feature vector size per node
        objects: (n_objects_1, n_objects_2), number of objects from timepoint
                t and t+1 as tuple
    """
    assert (objects[0] + objects[1]) == node_features.shape[0], f"Node features and objects have shape mismatch"
    
    return torch.cdist(node_features[:objects[0]], node_features[objects[0]:], p=2)

class TripleBatchLoss(nn.Module):
    
    def __init__(self, margin=10):
        super().__init__()
        self.margin = margin
        
    def forward(self, node_features, links):
        
        distances = pairwise_distance(node_features, links.shape)
        
        link_indices = torch.nonzero(links, as_tuple=True)
        
        # this is the positive term of the loss, corresponding 
        linked_distances = distances[link_indices]
        #print("Linked distances: ", linked_distances)
        
        negative_terms = torch.ones_like(linked_distances)
        mask_pos = links > 0
        mask_neg = ~mask_pos
        pinf = torch.ones_like(distances) * float('inf')
        
        dist_self_removed = torch.where(mask_neg, distances, pinf)
        #print("Distance self removed: " , dist_self_removed)
        
        num_links = len(link_indices[0])
        #print("Num of links: ", num_links)
        
        # verify this later
        first_neg_term = torch.min(dist_self_removed, dim=0)[0][link_indices[1]]
        second_neg_term = torch.min(dist_self_removed, dim=1)[0][link_indices[0]]
        #print("Firs neg term: ", first_neg_term)
        #print("Second neg term: ", second_neg_term)
        
        loss = linked_distances - first_neg_term - second_neg_term + self.margin
        
        #print("Loss: ", loss)
        return torch.mean(torch.clamp(loss, 0))


class AffinityLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, affinity_true, affinity_scores_pred):
        affinity_scores_pred = affinity_scores_pred.unsqueeze(0)
        affinity_true = affinity_true.unsqueeze(0)
        affinity_permute = affinity_scores_pred.permute(0, 3, 1, 2)
        #print(affinity_true.shape, "<-- true shape")
        #print(affinity_permute.shape, "<--- predicted shape")
        loss = self.loss(affinity_permute, affinity_true)
        return loss


class TrackerLoss(nn.Module):
    
    def __init__(self, margin=10.0, weight=0.5):
        super().__init__()
        self.margin = margin
        self.weight = weight
        # initialize different losses
        self.affinity_loss = AffinityLoss()
        self.triple_batch_loss = TripleBatchLoss(margin = self.margin)
        
    def forward(self, affinity_scores, node_features, links):
        # affinity loss
        affi_loss = self.affinity_loss(links, affinity_scores)
        
        # triplet batch loss
        trip_loss = self.triple_batch_loss(node_features, links)
        
        total_loss = 2.0 * (1 - self.weight) * affi_loss + 2.0 * self.weight * (trip_loss)
        
        #print("Affinity loss: ", affi_loss, "Triplet loss: ", trip_loss)
         
        return total_loss


loss_dict = {
    "trackerLoss": TrackerLoss

}