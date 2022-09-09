import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCELoss(nn.Module):
    r""" 0.5 * BCE + DICE
    """
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)
        
        bce_loss = self.bce_loss(prediction_flat, target_flat)
        
        return bce_loss


class WeightedMSE(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, target, weight):
        
        diff = (prediction - target) / 5.0
        
        return torch.mean(torch.square(diff) * weight)

class ArcCosDotLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, target, weight, mask):
        eps = 1e-12
        denominator = torch.multiply(torch.linalg.norm(prediction, dim=1),
                                    torch.linalg.norm(target, dim=1)) + eps
        dot_product = (prediction[:, 0, :, :] * target[:, 0, :, :] + prediction[:, 1, :, :] * target[:, 1, :, :])
        phasediff = torch.acos(torch.clip(dot_product/denominator, -0.999999, 0.999999))/3.141549
        return torch.mean(torch.square(phasediff[mask]) * weight[mask])
    
def derivatives(x, device):
    sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    depth = x.size()[1]
    sobel_kernel_x = torch.tensor(sobelx, dtype=torch.float32).unsqueeze(0).expand(depth,1,3,3).to(device)
    sobel_kernel_y = torch.tensor(sobely, dtype=torch.float32).unsqueeze(0).expand(depth,1,3,3).to(device)

    dx = torch.nn.functional.conv2d(x, sobel_kernel_x, stride=1, padding=1, groups=x.size(1))
    dy = torch.nn.functional.conv2d(x, sobel_kernel_y, stride=1, padding=1, groups=x.size(1))

    return dy,dx


class DerivativeLoss(nn.Module):
    
    def __init__(self, gpu=None):
        super().__init__()
        if (gpu is not None) and torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(gpu))
        else:
            self.device = torch.device("cpu")
    
    def forward(self, prediction, target, weight, mask):
        prediction_dx, prediction_dy = derivatives(prediction, self.device)
        target_dx, target_dy = derivatives(target, self.device)
        
        difference_x = (prediction_dx - target_dx) / 5.
        difference_y = (prediction_dy - target_dy) / 5.
        
        L_x = torch.square(difference_x)
        L_y = torch.square(difference_y)
        
        return torch.mean((L_x[mask] + L_y[mask]) * weight[mask])


class NormLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, prediction, target, weight, mask):
        prediction_norm = torch.linalg.norm(prediction, dim=1, keepdim=False)/5.0
        target_norm = torch.linalg.norm(target, dim=1, keepdim=False)/5.0

        diff = (prediction_norm - target_norm)

        return torch.mean(torch.square(diff[mask]) * weight[mask])

class OmniLoss(nn.Module):
    
    def __init__(self, gpu=None):
        super().__init__()
        # define all the criterions here
        
        # Weighted MSE loss between predcited and target flows
        self.loss1_criterion = WeightedMSE()
        
        # ArcCosDot Loss between predicted and target flows weighted
        self.loss2_criterion = ArcCosDotLoss()
        # derivative loss on the flows 
        self.loss3_criterion = DerivativeLoss(gpu=gpu)
        
        # Loss on the boundary
        self.loss4_criterion = nn.BCEWithLogitsLoss()
        
        # Norm loss between flows predicted and target
        self.loss5_criterion = NormLoss()
        
        # Weighted MSE loss between predicted and target distance fields
        self.loss6_criterion = WeightedMSE()
        
        # derivative loss on the distance field
        self.loss7_criterion = DerivativeLoss(gpu=gpu)
    
    # see that all tensors are in the same device
    def forward(self, prediction, target):
        # Target dims [N, 8, H, W] 
        # 
        flows_target = target[:, 2:4]
        dist_target = target[:, 1]
        boundary_target = target[:, 5]
        cellmask_target = (dist_target > 0).bool()
        weights_target = target[:, 7]
        
        # Prediction dims [N, 4, H, W] (flow_y, flow_x, dist_field, boundary)
        flows_prediction = prediction[:, :2] # first two indices are flow_y, flow_x
        boundary_prediction = prediction[:, 3]
        dist_prediction = prediction[:, 2]
        
        
        # stack two weigths_target to multiply both prediction and targets
        weights_target_twice = torch.stack((weights_target, weights_target), dim=1)
        cellmask_target_twice = torch.stack((cellmask_target, cellmask_target), dim=1)
        
        # 7 losses as in the omnipose paper/code
        # flows MSE loss
        loss1 = 10. * self.loss1_criterion(flows_prediction, flows_target, weights_target_twice)
        # flows arcCosDotLoss 
        loss2 = self.loss2_criterion(flows_prediction, flows_target,
                                     weights_target, cellmask_target)
        # Derivating loss between flows
        loss3 = self.loss3_criterion(flows_prediction, flows_target,
                                    weights_target_twice, cellmask_target_twice) / 10.
        # logits loss on boundary
        loss4 = 2. * self.loss4_criterion(boundary_prediction, boundary_target)
        # Norm loss on the flows
        loss5 = 2. * self.loss5_criterion(flows_prediction, flows_target,
                                          weights_target, cellmask_target)
        # dist MSE loss
        loss6 = 2. * self.loss6_criterion(dist_prediction, dist_target, weights_target)
        
        # Derivative loss between distance fields
        loss7 = self.loss7_criterion(dist_prediction.unsqueeze(1), dist_target.unsqueeze(1),
                                     weights_target.unsqueeze(1), cellmask_target.unsqueeze(1)) / 10.
        
        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7




class WeightedUnetLoss(nn.Module):
	"""
	Custom loss function for Unet BCE + DICE + weighting
	"""

	def __init__(self):
		super(WeightedUnetLoss, self).__init__()

	def forward(self, output, target, weights):

		output = torch.sigmoid(output)

		batch_size = target.shape[0]


		target_weighted = target - weights
	
		output_reshaped = output.view(batch_size, -1)
		target_weighted_reshaped = target_weighted.view(batch_size, -1)



		bce_loss = F.binary_cross_entropy(output_reshaped, target_weighted_reshaped)

		target_reshaped = target.view(batch_size, -1)

		intersection = (output_reshaped * target_reshaped)
		dice_per_image = 2. * (intersection.sum(1)) / (output_reshaped.sum(1) + target_reshaped.sum(1))
		dice_batch_loss = 1 - dice_per_image.sum() / batch_size
		print(dice_batch_loss.item(), bce_loss.item())
		return 0.5 * bce_loss  + 2.0 *dice_batch_loss
	#return bce_loss


class WeightedUnetLossExact(nn.Module):
	"""
	Custom loss function that implements the exact formulation of weighted BCE in 
	from the U-net paper https://arxiv.org/pdf/1505.04597.pdf
	"""

	def __init__(self):
		super(WeightedUnetLossExact, self).__init__()

	def forward(self, output, target, weights):
		
		output = torch.sigmoid(output)

		batch_size = target.shape[0]

		# calculate intersection over union
		target_reshaped = target.view(batch_size, -1)
		output_reshaped = output.view(batch_size, -1)
		intersection = (output_reshaped * target_reshaped)
		dice_per_image = 2. * (intersection.sum(1)) / (output_reshaped.sum(1) + target_reshaped.sum(1))
		dice_batch_loss = 1 - dice_per_image.sum() / batch_size

		# weighted cross entropy function

		weights_reshaped = weights.view(batch_size, -1)  + 1.0
		bce_loss = weights_reshaped * F.binary_cross_entropy(output_reshaped, target_reshaped, reduction='none')

		weighted_entropy_loss = bce_loss.mean()		
		print(dice_batch_loss.item(), weighted_entropy_loss.item())
		return dice_batch_loss  + 0.5 * weighted_entropy_loss



class UnetLoss(nn.Module):
	"""
	Custom loss function, for Unet, BCE + DICE
	"""
	def __init__(self):
		super(UnetLoss, self).__init__()
		self.bce_loss = nn.BCELoss()

	def forward(self, output, target):

		output = torch.sigmoid(output)
		output_flat = output.view(-1)
		target_flat = target.view(-1)

		bce_loss = self.bce_loss(output_flat, target_flat)

		batch_size = target.shape[0]

		output_dice = output.view(batch_size, -1)
		target_dice = target.view(batch_size, -1)

		intersection = (output_dice * target_dice)
		dice_per_image = 2. * (intersection.sum(1)) / (output_dice.sum(1) + target_dice.sum(1))

		dice_batch_loss = 1 - dice_per_image.sum() / batch_size
		print(dice_batch_loss.item(), bce_loss.item())
		return 0.5 * bce_loss + dice_batch_loss
		#return bce_loss


loss_dict = {
    "BCE": BCELoss,
    "Omni": OmniLoss
}

