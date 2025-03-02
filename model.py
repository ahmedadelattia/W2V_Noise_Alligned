import torch
import torch.nn as nn
from typing import Optional
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput

class Wav2Vec2ForDualInputPreTraining(Wav2Vec2ForPreTraining):
    """
    A subclass of Wav2Vec2ForPreTraining that supports dual inputs:
      - Noisy audio is processed through the full model (feature extractor → encoder).
      - If clean audio is provided, it is used solely for quantization (to compute
        contrastive and diversity losses) and to compute a consistency loss with the noisy features.
      - Otherwise, quantization uses the noisy encoder output.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.consistency_loss_weight = 1.0
        self.diversity_loss_weight = 0.1   # if you want to override parent's weight, adjust here
        self.feature_l2_penalty_weight = 10.0
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        input_values_noisy: torch.Tensor,
        attention_mask_noisy: Optional[torch.Tensor] = None,
        input_values_clean: Optional[torch.Tensor] = None,
        attention_mask_clean: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None,
        sampled_negative_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 1. Process the NOISY input normally
        #    (Using parent's functions for feature extraction and encoding)
        noisy_features = self.wav2vec2.feature_extractor(input_values_noisy)
        noisy_hidden_states, _ = self.wav2vec2.encoder(noisy_features, attention_mask=attention_mask_noisy)
        
        # 2. Compute quantized targets:
        #    If clean input is provided, use the clean features for quantization;
        #    otherwise, fall back to the noisy hidden states.
        if input_values_clean is not None:
            clean_features = self.wav2vec2.feature_extractor(input_values_clean)
            # Use clean features for quantization
            quantized_targets, codebook_loss, perplexity = self.wav2vec2.quantizer(clean_features)
            # Consistency loss: force noisy features to be close to clean features
            consistency_loss = self.mse_loss(noisy_features, clean_features)
        else:
            quantized_targets, codebook_loss, perplexity = self.wav2vec2.quantizer(noisy_hidden_states)
            consistency_loss = 0.0
        
        # 3. Compute contrastive loss.
        #    We assume the parent’s implementation has an internal function to compute contrastive loss.
        #    (If not publicly exposed, you may need to copy that logic from the parent’s forward.)
        contrastive_loss = self._compute_contrastive_loss(
            hidden_states=noisy_hidden_states,
            quantized_targets=quantized_targets,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        )
        
        # 4. Compute diversity loss.
        #    Again, using the parent’s internal function that operates on the quantized targets.
        diversity_loss = self._compute_diversity_loss(
            quantized_targets=quantized_targets,
            codebook_prob=codebook_loss,  # assuming parent's diversity loss uses these values
        )
        
        # 5. Optionally, add a feature L2 penalty on the final hidden states.
        feature_l2_penalty = noisy_hidden_states.pow(2).mean()
        
        # 6. Combine all losses.
        total_loss = (
            contrastive_loss 
            + self.config.diversity_loss_weight * diversity_loss
            + self.consistency_loss_weight * consistency_loss
            + self.feature_l2_penalty_weight * feature_l2_penalty
        )
        
        return Wav2Vec2ForPreTrainingOutput(
            loss=total_loss,
            logits=noisy_hidden_states,
            hidden_states=noisy_hidden_states,
            attentions=None,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )