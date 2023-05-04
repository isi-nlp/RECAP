import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalTypeEmbedding(nn.Module):
    
    def __init__(self, d_model=768, dropout=0.1, max_len=512):
        super().__init__()
        embed_dim = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.positional_embedding = nn.Parameter(torch.Tensor(max_len, embed_dim))
        nn.init.normal_(self.positional_embedding, mean=0, std=embed_dim ** -0.5)
        self.type_embedding = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.type_embedding, mean=0, std=embed_dim ** -0.5)
    
    def forward(self, x):
        bsz, seq_len, _ = x.size()
        assert seq_len % 2 == 0
        x += self.positional_embedding[:seq_len//2].repeat_interleave(2, dim=0)
        x += self.type_embedding.repeat(seq_len//2, 1)
        
        return self.dropout(x)


class Retriever(nn.Module):
    def __init__(self, model_name, nhead=12, use_gold_tgt_rep=False):
        super(Retriever, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=nhead, batch_first=True)
        layer_norm = nn.LayerNorm(768)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=layer_norm).cuda()
        self.positional_type_embedding = PositionalTypeEmbedding().cuda()

        self.src_encoder = AutoModel.from_pretrained(model_name)
        self.tgt_encoder = None if use_gold_tgt_rep else AutoModel.from_pretrained(model_name)
        self.tgt_proj = None if not use_gold_tgt_rep else nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)

        for layer in self.transformer_encoder.layers:
            for name, child in layer.named_children():
                if name == "self_attn":
                    child._reset_parameters()
                elif "dropout" not in name:
                    child.reset_parameters()
    
    def forward(
        self,
        srcs_ids,
        srcs_attention_mask,
        tgts_ids,
        tgts_attention_mask,
        labels,
    ):
        bsz, seq_num, seq_len = srcs_ids.size()

        src_reps = self.src_encoder(
            input_ids=srcs_ids.reshape(bsz*seq_num, -1), 
            attention_mask=srcs_attention_mask.reshape(bsz*seq_num, -1)
        ).last_hidden_state.mean(1).reshape(bsz, seq_num, -1)

        if self.tgt_encoder is None:
            tgt_reps = self.tgt_proj(labels)
        else:
            tgt_reps = self.tgt_encoder(
                input_ids=tgts_ids.reshape(bsz*seq_num, -1), 
                attention_mask=tgts_attention_mask.reshape(bsz*seq_num, -1)
            ).last_hidden_state.mean(1).reshape(bsz, seq_num, -1)

        transformer_input = torch.cat([src_reps, tgt_reps], dim=-1).reshape(bsz, seq_num*2, -1)
        transformer_input = self.positional_type_embedding(transformer_input)
        causal_mask = generate_square_subsequent_mask(seq_num*2, transformer_input.device)
        transformer_output = self.transformer_encoder(transformer_input, mask=causal_mask)
        transformer_output = self.out_proj(transformer_output[: ,::2, :])

        transformer_output = transformer_output[:, -10:, :]
        labels = labels[:, -10:, :]

        loss_fct = nn.CosineEmbeddingLoss()
        loss = loss_fct(
            transformer_output.reshape(bsz*seq_num//10, -1), 
            labels.reshape(bsz*seq_num//10, -1), 
            target=torch.ones(bsz*seq_num//10).to(transformer_output.device)
        )

        return SequenceClassifierOutput(
            loss=loss,
            logits=transformer_output,
            hidden_states=None,
        )