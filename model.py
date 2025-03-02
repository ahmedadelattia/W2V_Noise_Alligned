'''
Date: Mar 2, 2025
Author: Ahmed Attia
Adapted from: https://huggingface.co/transformers/v4.7.0/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html
'''


import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Model, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput, Wav2Vec2BaseModelOutput
from dataclasses import dataclass
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.tensor:
    """
    Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
    ASR <https://arxiv.org/abs/1904.08779>`__.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + torch.rand((1,)).item())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones((batch_size, sequence_length - (mask_length - 1)), device=device)

    # get random indices to mask
    spec_aug_mask_idxs = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = (
        spec_aug_mask_idxs.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = spec_aug_mask.scatter(1, spec_aug_mask_idxs, True)

    return spec_aug_mask

@dataclass
class Wav2Vec2ForDualInputBaseModelOutput(Wav2Vec2BaseModelOutput):
    """
    Output type of :class:`~transformers.Wav2Vec2BaseModelOutput`, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    extract_noisy_features: torch.FloatTensor = None
    extract_clean_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ForDualInputModel(Wav2Vec2Model):
    
    def forward(
        self,
        input_noisy_values,
        input_clean_values = None,
        attention_mask=None,
        output_attentions=None,
        mask_time_indices=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_noisy_features = self.feature_extractor(input_noisy_values)
        extract_noisy_features = extract_noisy_features.transpose(1, 2)
        hidden_noisy_states, extract_noisy_features = self.feature_projection(extract_noisy_features)

        if input_clean_values is not None:
            extract_clean_features = self.feature_extractor(input_clean_values)
            extract_clean_features = extract_clean_features.transpose(1, 2)
            hidden_clean_states, extract_clean_features = self.feature_projection(extract_clean_features)

        else:
            extract_clean_features = extract_noisy_features
        
        if attention_mask is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            attention_mask = torch.zeros(
                extract_noisy_features.shape[:2], dtype=extract_noisy_features.dtype, device=extract_noisy_features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            attention_mask[
                (torch.arange(attention_mask.shape[0], device=extract_noisy_features.device), output_lengths - 1)
            ] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()



        if mask_time_indices is not None:  # apply SpecAugment along time axis with given indices
            hidden_noisy_states[mask_time_indices] = self.masked_spec_embed.to(hidden_noisy_states.dtype)

        hidden_noisy_states = self._mask_hidden_states(hidden_noisy_states)

        encoder_outputs = self.encoder(
            hidden_noisy_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, extract_noisy_features, extract_clean_features) + encoder_outputs[1:]

        return Wav2Vec2ForDualInputBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_noisy_features=extract_noisy_features,
            extract_clean_features=extract_clean_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class Wav2Vec2ForDualInputPreTraining(Wav2Vec2ForPreTraining):
    
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2ForDualInputModel(config)
        
    #unchanged. Here for inheritance issues
    @staticmethod
    def _sample_negatives(features: torch.FloatTensor, num_negatives: int):
        """
        Sample `num_negatives` vectors from feature vectors.
        """
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
            )

        features = features.view(-1, hidden_size)  # BTC => (BxT)C

        with torch.no_grad():
            # get `num_negatives` random vector indices from the same utterance
            sampled_negative_indices = torch.randint(
                low=0,
                high=sequence_length - 1,
                size=(batch_size, num_negatives * sequence_length),
                device=features.device,
            )

            # generate indices of the positive vectors themselves, repeat them `num_negatives` times
            feature_indices = (
                torch.arange(sequence_length, device=features.device)[:, None]
                .expand(sequence_length, num_negatives)
                .flatten()
            )

            # avoid sampling the same positive vector, but keep the distribution uniform
            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        # correct for batch size
        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        # take negative vectors from sampled indices
        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(batch_size, sequence_length, num_negatives, hidden_size).permute(
            2, 0, 1, 3
        )

        return sampled_negatives

    def forward(
        self,
        input_noisy_values,
        input_clean_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, raw_sequence_length = input_noisy_values.shape
        sequence_length = self._get_feat_extract_output_lengths(raw_sequence_length)
        mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2, device=input_noisy_values.device)


        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_noisy_values,
            input_clean_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            mask_time_indices=mask_time_indices,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # last_hidden_state: torch.FloatTensor = None
    # extract_noisy_features: torch.FloatTensor = None
    # extract_clean_features: torch.FloatTensor
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_clean_features = self.dropout_features(outputs[2])
        quantized_features, codevector_perplexity = self.quantizer(extract_clean_features, mask_time_indices)
        quantized_features = self.project_q(quantized_features)

        loss = None
        if self.training:
            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            negative_quantized_features = self._sample_negatives(quantized_features, self.config.num_negatives)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            preds = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
            contrastive_loss = nn.functional.cross_entropy(preds.float(), target, reduction="sum")

            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors

            # 8. compute consistency loss
            consistency_loss = nn.functional.mse_loss(outputs[1], outputs[2])
            
            # 9. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss + self.config.consistency_loss_weight * consistency_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
