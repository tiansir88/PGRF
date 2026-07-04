import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable
from timm.models.layers import Mlp
from fairseq_signals_backbone.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel, Wav2Vec2CMSCConfig
from lightning import LightningModule
from transformers import PreTrainedModel
from .configuration_MELP_Encoder import MELPEncoderConfig


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    

class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ECGFMModel(LightningModule):
    def __init__(self, 
                 model_size: str = "small", # small by default
                 shared_emb_dim: int = 256,
                 embed_dim_caption: int = 768,
                 use_attentional_pool_contrast: bool = False,
                 use_attentional_pool_caption: bool = False,
                 n_queries_contrast: int = 10,
                 n_queries_caption: int = 128,
                 attn_pooler_heads: int = 8,
                 norm_layer: nn.Module = nn.LayerNorm,
                 proj: str = "linear",
                 drop: float = 0.,
                 proj_bias: bool = False,
                 num_leads: int = 12,
                 softmax_temperature: float = 0.1,
                 lambd: float = 0.0051,
                 *args,
                 **kwargs):
        
        """" Implementation of ECG-FM model.
        Using the Wave2Vec2 model as the ECG encoder: CNN + Transformer
        
        """
        super().__init__()
        self.save_hyperparameters()
        self.shared_emb_dim = shared_emb_dim
        self.num_leads = num_leads
        self.temperature = softmax_temperature

        if model_size == "small":
            self.encoder_embed_dim = 768
            self.encoder_attention_heads = 12
            self.encoder_layers = 8
            self.encoder_ffn_embed_dim = 3072
        elif model_size == "base":
            self.encoder_embed_dim = 768
            self.encoder_attention_heads = 12
            self.encoder_layers = 12
            self.encoder_ffn_embed_dim = 3072
        elif model_size == "large":
            self.encoder_embed_dim = 1024
            self.encoder_attention_heads = 16
            self.encoder_layers = 24
            self.encoder_ffn_embed_dim = 4096
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        print("Using ECG encoder with the following configuration:")
        print(f"encoder_embed_dim: {self.encoder_embed_dim}")
        print(f"encoder_attention_heads: {self.encoder_attention_heads}")
        print(f"encoder_layers: {self.encoder_layers}")
        print(f"encoder_ffn_embed_dim: {self.encoder_ffn_embed_dim}")
        
        self.init_ecg_encoder()

        self.embed_dim_caption = embed_dim_caption
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
    
        head_layers = OrderedDict()
        prev_chs = self.ecg_encoder.cfg.encoder_embed_dim
        if use_attentional_pool_contrast:
            scale = prev_chs ** -0.5
            self.attn_pool_contrast = AttentionalPooler(
                d_model=shared_emb_dim, 
                context_dim=prev_chs, 
                n_head=attn_pooler_heads, 
                n_queries=n_queries_contrast)
            self.ln_contrast = norm_layer(shared_emb_dim)
            self.proj_contrast = nn.Parameter(scale * torch.randn(shared_emb_dim, shared_emb_dim))
        else:
            assert proj, 'projection layer needed if not using attentional pooling.'
            # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
            if proj == 'linear':
                head_layers['drop'] = nn.Dropout(drop)
                head_layers['proj'] = nn.Linear(prev_chs, shared_emb_dim, bias=proj_bias)
            elif proj == 'mlp':
                head_layers['mlp'] = Mlp(prev_chs, 2 * shared_emb_dim, shared_emb_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

        if use_attentional_pool_caption:
            self.attn_pool_caption = AttentionalPooler(
                d_model=embed_dim_caption, context_dim=prev_chs, n_head=attn_pooler_heads, n_queries=n_queries_caption)
            self.ln_caption = norm_layer(embed_dim_caption)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.bn = nn.BatchNorm1d(768, affine=False)
        self.lambd = lambd

    def init_ecg_encoder(self):
        # Here we define Wav2Vec2CMSC model as the ECG encoder
        cfg = Wav2Vec2CMSCConfig(
            apply_mask = True,
            mask_prob = 0.65,
            quantize_targets = True,
            final_dim = 256,
            dropout_input = 0.1,
            dropout_features = 0.1,
            feature_grad_mult = 0.1,
            encoder_embed_dim = self.encoder_embed_dim,
            encoder_attention_heads = self.encoder_attention_heads,
            in_d = 12,
            encoder_layers = self.encoder_layers,
            encoder_ffn_embed_dim = self.encoder_ffn_embed_dim
        )
        self.ecg_encoder = Wav2Vec2CMSCModel(cfg)

    def _global_pool(self, x):
        return torch.mean(x, dim=1)
        
    @torch.no_grad()
    # only used for finetune ...
    def ext_ecg_emb(self, ecg, normalize=False):
        assert ecg.dim() == 3, "Input tensor must be 3D"

        ecg_out = self.ecg_encoder(source=ecg, mask=False, features_only=True)
        features = ecg_out["x"]

        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(features)
            pooled = self.ln_contrast(pooled)
            pooled = torch.mean(pooled, dim=1)
        else:
            pooled = self._global_pool(features)

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled

    def _encode_ecg(self, ecg):
        assert ecg.dim() == 3, "Input tensor must be 3D"
        ecg_out = self.ecg_encoder(source=ecg, mask=False, features_only=True)
        # features = self.ecg_encoder.get_features(net_output=ecg_out, aggregate=False)
        # results after CNN-Transformer
        features = ecg_out["x"]

        if self.use_attentional_pool_contrast:
            # hierarchical pooling
            pooled = self.attn_pool_contrast(features)
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast.unsqueeze(0)
            pooled_beat = pooled.clone()
            pooled = torch.mean(pooled, dim=1)
        else:
            pooled = self._global_pool(features)
            pooled = self.head(features)

        tokens = None
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(features)
            tokens = self.ln_caption(tokens)
        else:
            tokens = None

        return pooled, pooled_beat, tokens
    
    def encode_ecg(self, ecg):
        ecg_latent, _, _ = self._encode_ecg(ecg)
        return ecg_latent


class MELPEncoderModel(PreTrainedModel):
    config_class = MELPEncoderConfig

    def __init__(self, config: MELPEncoderConfig):
        super().__init__(config)

        self.ecg_encoder = ECGFMModel(
            model_size=config.model_size,
            shared_emb_dim=config.shared_emb_dim,
            embed_dim_caption=config.embed_dim_caption,
            use_attentional_pool_contrast=config.use_attentional_pool_contrast,
            use_attentional_pool_caption=config.use_attentional_pool_caption,
            n_queries_contrast=config.n_queries_contrast,
            n_queries_caption=config.n_queries_caption,
            attn_pooler_heads=config.attn_pooler_heads,
            proj=config.proj,
            drop=config.drop,
            proj_bias=config.proj_bias,
            num_leads=config.num_leads,
        )
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        proj_ecg_emb, ecg_beat_emb, ecg_token_emb = self.ecg_encoder._encode_ecg(tensor)

        return {
            "proj_ecg_emb": proj_ecg_emb,
            "ecg_beat_emb": ecg_beat_emb,
            "ecg_token_emb": ecg_token_emb
        }


