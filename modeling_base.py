# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import ModelOutput


class BaseModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class Classifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_labels,
        dropout=0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BaseModel(PreTrainedModel):
    def __init__(self, hparams, num_labels):
        config = AutoConfig.from_pretrained(
            hparams.pretrained_model_name, num_labels=num_labels
        )
        super().__init__(config)
        setattr(
            self,
            self.config.model_type,
            AutoModel.from_pretrained(
                hparams.pretrained_model_name, config=self.config
            ),
        )
        self.classifier = Classifier(
            self.config.hidden_size,
            num_labels,
            dropout=hparams.classifier_dropout_prob,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            assert input_ids.dim() == 2  # batch x len

        encoder_outputs = getattr(self, self.config.model_type)(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        features = encoder_outputs.last_hidden_state[:, 0]  # equiv. to [CLS]

        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return BaseModelOutput(loss=loss, logits=logits)
