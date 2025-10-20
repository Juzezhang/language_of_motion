import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Simple wrapper class that passes through the commitment loss.
    This is used to standardize the loss interface.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit
    


class VAELosses(BaseLosses):
    """
    Loss calculation class for different training stages.
    Handles various loss components like reconstruction, commitment.
    """

    def __init__(self, cfg, stage, **kwargs):
        # Save the current training stage
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        # Initialize loss components and their parameters
        losses = []
        params = {}
        
        # Configure losses based on training stage
        # For vector quantization stages

        if cfg.DATASET.motion_representation == 'rotation':
            # For separate rotation representation: handle each body part separately
            # Reconstruction losses for different body parts
            losses.append("recons-upper_loss") 
            params['recons-upper_loss'] = 1.0 
            losses.append("recons-lower_loss") 
            params['recons-lower_loss'] = 1.0 
            losses.append("recons-global_loss") 
            params['recons-global_loss'] = 1.0
            losses.append("recons-face_loss")
            params['recons-face_loss'] = 1.0
            losses.append("recons-hand_loss")
            params['recons-hand_loss'] = 1.0
            losses.append("recons-global_loss")
            params['recons-global_loss'] = 1.0
            # Commitment losses for different body parts
            losses.append("commit-upper_loss")
            params['commit-upper_loss'] = 1.0
            losses.append("commit-lower_loss")
            params['commit-lower_loss'] = 1.0
            losses.append("commit-global_loss")
            params['commit-global_loss'] = 1.0
            losses.append("commit-face_loss")
            params['commit-face_loss'] = 1.0
            losses.append("commit-hand_loss")
            params['commit-hand_loss'] = 1.0
        else:
            # For unified representation: one loss for the entire body
            losses.append("recons_loss") 
            params['recons_loss'] = 1.0 
            losses.append("commit_loss")
            params['commit_loss'] = 1.0

        # Map loss types to their corresponding loss functions
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                # Select reconstruction loss type based on configuration
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in [
                'commit', 'loss'
            ]:
                # Use the CommitLoss wrapper for these types
                losses_func[loss] = CommitLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        # Initialize the base class with configured losses and parameters
        super().__init__(cfg, losses, params, losses_func,
                         **kwargs)

    def update(self, rs_set):
        '''
        Calculate and update all loss components based on the current results set.
        
        Args:
            rs_set: Dictionary containing current model outputs and targets
            
        Returns:
            total: The combined loss value for backpropagation
        '''
        total: float = 0.0
        # For vector quantization stages: sum all losses with "loss" in their name
        for key, value in rs_set.items():
            if "loss" in key:
                # Use _update_loss method to update each loss value
                total += self._update_loss(key, None, None, precomputed_val=value)
            
        # Update accumulated loss statistics
        self.total += total.detach()
        self.count += 1

        return total


class GPTLosses(BaseLosses):
    """
    Loss calculation class for different training stages.
    Handles various loss components like reconstruction, commitment, and language modeling.
    """

    def __init__(self, cfg, stage, **kwargs):
        # Save the current training stage
        self.stage = stage
        # recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        recons_loss = cfg.ABLATION.RECONS_LOSS

        # Initialize loss components and their parameters
        losses = []
        params = {}
        # For language model training stages
        losses.append("gpt_loss")
        params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS

        # Map loss types to their corresponding loss functions
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                # Select reconstruction loss type based on configuration
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in [
                'commit', 'loss', 'gpt', 'm2t2m', 't2m2t'
            ]:
                # Use the CommitLoss wrapper for these types
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                # Use cross entropy for classification/language modeling
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        # Initialize the base class with configured losses and parameters
        super().__init__(cfg, losses, params, losses_func,
                         **kwargs)

    def update(self, rs_set):
        '''
        Calculate and update all loss components based on the current results set.
        
        Args:
            rs_set: Dictionary containing current model outputs and targets
            
        Returns:
            total: The combined loss value for backpropagation
        '''
        total: float = 0.0

        # For language model training: use the loss directly from model outputs
        total += self._update_loss("gpt_loss", rs_set['outputs'].loss, rs_set['outputs'].loss, precomputed_val=None)

        # Update accumulated loss statistics
        self.total += total.detach()
        self.count += 1

        return total
