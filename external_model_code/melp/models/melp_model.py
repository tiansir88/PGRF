'''
Our proposed approach: MELP
Contributions:
1. Coca-style pretraining approach.
2. SOTA unimodal pretraining.
3. Local-to-Global cross-modal learning.
'''
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import ot
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import create_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from melp.models.merl_model import MERLModel
from melp.backbone.transformer import  (
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from melp.utils.openclip_loss import CoCaLoss
from melp.models.ecgfm_model import ECGFMModel
from melp.backbone.transformer import AttentionalPooler
from melp.backbone.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from melp.backbone import modeling_finetune
from melp.paths import ECGFM_PATH
try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StopStringCriteria,
        EosTokenCriteria,
        StoppingCriteriaList
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


@dataclass
class MultimodalConfig:
    width: int = 768
    context_length: int = 128
    heads: int = 8
    n_queries: int = 256
    layers: int = 6
    ls_init_value: Optional[float] = None
    quick_gelu: bool = False


def _token_to_tensor(token_id, device: str = "cpu") -> torch.Tensor:
    if not isinstance(token_id, torch.Tensor):
        if isinstance(token_id, int):
            token_id = [token_id]
        token_id = torch.tensor(token_id, device=device)
    return token_id


class BeatAlignmentModule(nn.Module):
    ''' Maybe change to gloria loss '''
    def __init__(self, ot_reg=0.1, ot_tau=0.5):
        super().__init__()
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
    
    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
        
    @staticmethod
    def attention_fn(query, context, temp1):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        batch_size, queryL = query.size(0), query.size(2)
        ih, iw = context.size(2), context.size(3)
        sourceL = ih * iw

        # --> batch x sourceL x ndf
        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)

        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        weightedContext = torch.bmm(context, attnT)

        return weightedContext, attn.view(batch_size, -1, ih, iw)

    def forward(self, ecg_embs, sent_embs, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):
        '''
        From GLoRIA
        ecg_embs: (B, L, D)
        sent_embs: (B, L, D)
        '''
        batch_size = ecg_embs.size(0)

        att_maps = []
        similarities = []
        for i in range(batch_size):
            sent = sent_embs[i].transpose(0, 1).unsqueeze(0).contiguous() # 1, 256, 2
            sent_num = sent.size(2)
            sent = sent.repeat(batch_size, 1, 1) # B, 256, 2
            context = ecg_embs.transpose(1, 2).unsqueeze(2).contiguous() # B, 256, 1, 128

            weiContext, attn = self.attention_fn(
                sent, context, temp1
            )  # [B, 256, 2], [B, 256, 1, 128]

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 19, 19]

            sent = sent.transpose(1, 2).contiguous() 
            weiContext = weiContext.transpose(1, 2).contiguous() 

            sent = sent.view(batch_size * sent_num, -1) 
            weiContext = weiContext.view(batch_size * sent_num, -1) 

            row_sim = self.cosine_similarity(sent, weiContext)
            row_sim = row_sim.view(batch_size, sent_num)  # [48, 25]

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]

        labels = torch.arange(batch_size).type_as(ecg_embs).long()
        loss0 = F.cross_entropy(similarities, labels)  # labels: arange(batch_size)
        loss1 = F.cross_entropy(similarities1, labels)

        return (loss0 + loss1) / 2, att_maps


class CustomResNet18(nn.Module):
    def __init__(self, 
                 norm_layer: nn.Module = nn.LayerNorm,
                 proj_out: int = 256,
                 shared_emb_dim: int = 256,
                 embed_dim_caption: int = 768,
                 attn_pooler_heads: int = 8,
                 n_queries_caption: int = 128,
                 n_queries_contrast: int = 10,
                 ):
        super().__init__()

        self.model = ResNet18()
        self.downconv = nn.Conv1d(in_channels=512, out_channels=proj_out, kernel_size=1)

        prev_chs = 256
        scale = prev_chs ** -0.5
        self.attn_pool_caption = AttentionalPooler(
            d_model=embed_dim_caption, context_dim=prev_chs, n_head=attn_pooler_heads, n_queries=n_queries_caption)
        self.ln_caption = norm_layer(embed_dim_caption)
    
        self.attn_pool_contrast = AttentionalPooler(
            d_model=shared_emb_dim, 
            context_dim=prev_chs, 
            n_head=attn_pooler_heads, 
            n_queries=n_queries_contrast)
        self.ln_contrast = norm_layer(shared_emb_dim)
        self.proj_contrast = nn.Parameter(scale * torch.randn(shared_emb_dim, shared_emb_dim))

    def _encode_ecg(self, ecg):
        ecg_emb = self.model(ecg)
        ecg_emb = self.downconv(ecg_emb)   # bz, 256, 313
        ecg_emb = ecg_emb.permute(0, 2, 1) # bz, 313, 256

        pooled = self.attn_pool_contrast(ecg_emb)
        pooled = self.ln_contrast(pooled)
        pooled = pooled @ self.proj_contrast.unsqueeze(0)
        pooled_beat = pooled.clone()
        pooled = torch.mean(pooled, dim=1)
    
        tokens = self.attn_pool_caption(ecg_emb)
        tokens = self.ln_caption(tokens)

        return pooled, pooled_beat, tokens 

    @torch.no_grad()
    def ext_ecg_emb(self, ecg, normalize=False):
        ecg_emb = self.model(ecg)
        ecg_emb = self.downconv(ecg_emb)   # bz, 256, 313
        ecg_emb = ecg_emb.permute(0, 2, 1) # bz, 313, 256

        pooled = self.attn_pool_contrast(ecg_emb)
        pooled = self.ln_contrast(pooled)
        # pooled = pooled @ self.proj_contrast.unsqueeze(0)
        # pooled_beat = pooled.clone()
        pooled = torch.mean(pooled, dim=1)

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
            
        # tokens = self.attn_pool_caption(ecg_emb)
        # tokens = self.ln_caption(tokens)

        return pooled


class MELPModel(MERLModel):
    def __init__(self, 
                 ecg_encoder_name: str = "ecgfm",
                 ecg_encoder_weight: str = "",
                 text_encoder_name: str = "ncbi/MedCPT-Query-Encoder",
                 val_dataset_list: List = ["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
                                           "icbeb", "chapman"],
                 max_seq_len: int = 128,
                 n_queries_contrast: int = 13,
                 clip_loss_weight: float = 1.0,
                 caption_loss_weight: float = 1.0,
                 local_loss_weight: float = 1.0,
                 shared_emb_dim: int = 256,
                 num_leads: int = 12,
                 num_freeze_layers: int = 6,
                 init_logit_scale: float = np.log(1 / 0.07),
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 *args,
                 **kwargs):
        
        self.n_queries_contrast = n_queries_contrast
        self.ecg_encoder_weight = ecg_encoder_weight
        self.shared_emb_dim = shared_emb_dim
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.local_loss_weight = local_loss_weight
        self.max_seq_len = max_seq_len

        super().__init__(ecg_encoder_name=ecg_encoder_name,
                         text_encoder_name=text_encoder_name,
                         val_dataset_list=val_dataset_list,
                         shared_emb_dim=shared_emb_dim,
                         num_leads=num_leads,
                         num_freeze_layers=num_freeze_layers,
                         init_logit_scale=init_logit_scale,
                         lr=lr,
                         weight_decay=weight_decay,
                         *args,
                         **kwargs)
        self.save_hyperparameters()

        if self.text_encoder_name in ["ncbi/MedCPT-Query-Encoder", "fuyingw/heart_bert"]:
            self.sent_proj = nn.Linear(768, self.shared_emb_dim)
        else:
            raise NotImplementedError

        # build text decoder
        multimodal_cfg = MultimodalConfig()
        act_layer = QuickGELU if multimodal_cfg.quick_gelu else nn.GELU
        norm_layer = LayerNorm
        self.text_decoder = MultimodalTransformer(
            context_length=multimodal_cfg.context_length,
            width=multimodal_cfg.width,
            heads=multimodal_cfg.heads,
            layers=multimodal_cfg.layers,
            ls_init_value=multimodal_cfg.ls_init_value,
            output_dim=self.lm_model.config.vocab_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.uot_loss = BeatAlignmentModule()

    def init_ecg_encoder(self):
        if self.ecg_encoder_name == "ecgfm":
            if self.ecg_encoder_weight:
                print("Loading ECGFM model from checkpoint {}".format(self.ecg_encoder_weight))
                self.ecg_encoder = ECGFMModel(
                                    use_attentional_pool_contrast=True,
                                    use_attentional_pool_caption=True,
                                    n_queries_caption=128,
                                    model_size="small"
                                    )
                ckpt = torch.load(self.ecg_encoder_weight)["state_dict"]
                new_ckpt = dict()
                for k, v in ckpt.items():
                    if "attn_pool_contrast" not in k:
                        new_ckpt[k] = v
                unloaded_keys = self.ecg_encoder.load_state_dict(new_ckpt, strict=False)
                print(f"Unloaded keys: {unloaded_keys}")
            else:
                # We use 8 layers of transformer layers by default ...
                self.ecg_encoder = ECGFMModel(
                    use_attentional_pool_contrast=True,
                    use_attentional_pool_caption=True,
                    n_queries_caption=128,
                    n_queries_contrast=self.n_queries_contrast,
                    model_size="small"
                )
        elif self.ecg_encoder_name == "resnet18":
            self.ecg_encoder = CustomResNet18()

        else:
            raise NotImplementedError

    def init_text_encoder(self):
        if self.text_encoder_name in ["ncbi/MedCPT-Query-Encoder", "fuyingw/heart_bert"]:
            self.lm_model = AutoModelForCausalLM.from_pretrained(
                self.text_encoder_name, is_decoder=True)

            # freeze layers
            for layer_idx in range(self.num_freeze_layers):
                for param in list(self.lm_model.bert.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False

            text_encoder_hidden_dim = 768
        else:
            raise NotImplementedError
        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(text_encoder_hidden_dim, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.text_encoder_name)
        self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.lm_model.resize_token_embeddings(len(self.tokenizer))
    
    def _tokenize(self, text):
        # manually add bos and sep token at the beginning and end of the text
        text = [f"[BOS] {t}" for t in text]
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=self.max_seq_len,
                                                            padding='max_length',
                                                            return_tensors='pt')
        # because orginal tokenizer have a cls token at the beginning
        # remove cls token and append at the endding
        tokenizer_output["input_ids"] = tokenizer_output["input_ids"][:, 1:]
        tokenizer_output["attention_mask"] = tokenizer_output["attention_mask"][:, 1:]
        return tokenizer_output
    
    def _encode_ecg(self, ecg, normalize: bool = True):
        proj_ecg_emb, ecg_beat_emb, ecg_token_emb = self.ecg_encoder._encode_ecg(ecg)

        if normalize:
            proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)

        return proj_ecg_emb, ecg_beat_emb, ecg_token_emb

    def encode_ecg(self, ecg, normalize=True, proj_contrast=True):
        if proj_contrast:
            ecg_latent, _, _ = self._encode_ecg(ecg, normalize=normalize)
        else:
            ecg_latent = self.ecg_encoder.forward_no_head(ecg, normalize=normalize)

        return ecg_latent

    def aggregate_sentence_emb(self, input_ids, token_embs):
        ''' Aggregate sentence embeddings from token embeddings '''
        special_ids = torch.tensor(self.tokenizer.all_special_ids).type_as(input_ids)
        batch_sent_embs = []
        for (input_ids_per_sample, token_embs_per_sample) in zip(input_ids, token_embs):
            # split base on .
            sep_pos = (input_ids_per_sample == 18).nonzero(as_tuple=True)[0] + 1
            sep_pos = torch.cat((torch.tensor([0]).type_as(sep_pos), sep_pos))
            sent_embs = []
            for i in range(len(sep_pos) - 1):
                sent_ids = input_ids_per_sample[sep_pos[i]:sep_pos[i+1]]
                sent_emb = token_embs_per_sample[sep_pos[i]:sep_pos[i+1]]
                sent_mask = ~torch.isin(sent_ids, special_ids)
                if sent_mask.sum() < 1:
                    continue
                sent_emb = sent_emb[sent_mask].mean(dim=0)
                sent_embs.append(sent_emb)

            if len(sent_embs) > 0:
                batch_sent_embs.append(torch.stack(sent_embs))
                
        return batch_sent_embs

    def _encode_text(self, input_ids, attention_mask, normalize=True, return_sent_emb=True):
        if self.text_encoder_name in ["ncbi/MedCPT-Query-Encoder", "fuyingw/heart_bert"]:
            input_ids = torch.cat((
                input_ids,
                self.tokenizer.cls_token_id * torch.ones(len(input_ids), 1, dtype=torch.long).type_as(input_ids)),
                dim=1)
            attention_mask = torch.cat((
                attention_mask,
                torch.ones(len(input_ids), 1, dtype=torch.long).type_as(input_ids)),
                dim=1)
            text_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = text_output.hidden_states[-1]
            text_emb = last_hidden_state[:, -1]
            token_emb = last_hidden_state[:, :-1]
            if return_sent_emb:
                sent_emb = self.aggregate_sentence_emb(input_ids, token_emb)
        else:
            raise NotImplementedError

        proj_text_emb = self.proj_t(text_emb)

        if normalize:
            proj_text_emb = F.normalize(proj_text_emb, dim=-1)

        if return_sent_emb:
            return proj_text_emb, token_emb, sent_emb
        else:
            return proj_text_emb, token_emb
    
    def encode_text(self, input_ids, attention_mask, normalize: bool = True, return_sent_emb: bool = True):
        if return_sent_emb:
            text_latent, _, _ = self._encode_text(input_ids, attention_mask, normalize=normalize, return_sent_emb=True)
        else:
            text_latent, _ = self._encode_text(input_ids, attention_mask, normalize=normalize, return_sent_emb=False)
        return text_latent

    @torch.no_grad()
    def ext_ecg_emb(self, ecg, normalize=False):
        # extract global ECG embedding
        pooled = self.ecg_encoder.ext_ecg_emb(ecg, normalize=normalize)

        return pooled
        
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        if self.text_encoder_name in ["ncbi/MedCPT-Query-Encoder", "*/heart_bert"]:
            text_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
            # using the CLS token as the global embedding
            text_emb = text_output.last_hidden_state[:, -1]
        else:
            raise NotImplementedError

        proj_text_emb = self.proj_t(text_emb)

        return proj_text_emb

    def forward_for_logits(self,
                           ecg: torch.Tensor,  
                           text: torch.Tensor,
                           ecg_latent: Optional[torch.Tensor] = None,
                           ecg_embs: Optional[torch.Tensor] = None,
                           output_labels: bool = True
                           ):
        
        if ecg_latent is None or ecg_embs is None:
            ecg_latent, _, ecg_embs = self._encode_ecg(ecg)
        if text is None:
            return {"ecg_latent": ecg_latent, "ecg_embs": ecg_embs}

        input_ids = text
        attention_mask = (input_ids != self.tokenizer.pad_token_type_id).long()
        text_latent, token_embs = self._encode_text(input_ids, attention_mask, return_sent_emb=False)

        if output_labels:
            # align text_embs and thus logits with labels for teacher-forcing caption loss
            token_embs = token_embs[:, :-1]
        # FIXME this isn't an ideal solution, would like to improve -RW
        labels: Optional[torch.Tensor] = input_ids[:, 1:] if output_labels else None

        # Forward text decoder
        logits = self.text_decoder(ecg_embs, token_embs)

        return {
            "ecg_latent": ecg_latent,
            "text_latent": text_latent,
            "logits": logits,
            "logit_scale": self.logit_scale.exp(),
            "labels": labels
        }

    def forward(self, 
                ecg: torch.Tensor,
                text: Optional[torch.Tensor] = None,
                ecg_latent: Optional[torch.Tensor] = None,
                ecg_embs: Optional[torch.Tensor] = None,
                output_labels: bool = True
                ):
        
        if ecg_latent is None or ecg_embs is None:
            ecg_latent, ecg_beat_embs, ecg_embs = self._encode_ecg(ecg)
        if text is None:
            return {"ecg_latent": ecg_latent, "ecg_embs": ecg_embs}

        if isinstance(text, torch.Tensor):
            input_ids = text
            attention_mask = (input_ids != self.tokenizer.pad_token_type_id).long()
        else:
            text_output = self._tokenize(text)
            input_ids = text_output.input_ids.type_as(ecg).long()
            attention_mask = text_output.attention_mask.type_as(ecg).long()
        text_latent, token_embs, sent_embs = self._encode_text(input_ids, attention_mask)

        if output_labels:
            # align text_embs and thus logits with labels for teacher-forcing caption loss
            token_embs = token_embs[:, :-1]
        # FIXME this isn't an ideal solution, would like to improve -RW
        labels: Optional[torch.Tensor] = input_ids[:, 1:] if output_labels else None

        # Forward text decoder
        # ecg_embs: bz, 128, 768
        # token_embs: bz, 126, 7768
        logits = self.text_decoder(ecg_embs, token_embs)

        sent_embs = [self.sent_proj(sent_emb) for sent_emb in sent_embs]
        out_dict = {
            "ecg_latent": ecg_latent,
            "text_latent": text_latent,
            "logits": logits,
            "logit_scale": self.logit_scale.exp(),
            "ecg_embs": ecg_beat_embs,
            "sent_embs": sent_embs
        }

        if labels is not None:
            out_dict["labels"] = labels

        return out_dict

    def shared_step(self, batch, batch_idx):
        
        # only used in the training step
        if (batch_idx % 1000 == 0) and (self.local_rank == 0):
            print(f"Generated reports in rank {torch.distributed.get_rank()}")
            ecg_out = self.generate(batch['ecg'][:4], seq_len=self.max_seq_len)
            print(self.tokenizer.batch_decode(ecg_out[:4], skip_special_tokens=True))
            print("Ground truth:")
            print(batch['report'][:4])

        output_dict = self(batch["ecg"], batch['report'])
        uot_loss, _ = self.uot_loss(output_dict["ecg_embs"], output_dict["sent_embs"])
        uot_loss *= self.local_loss_weight

        coca_loss = CoCaLoss(
            caption_loss_weight=self.caption_loss_weight,
            clip_loss_weight=self.clip_loss_weight,
            pad_id= self.tokenizer.pad_token_type_id,
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            use_horovod=False
        )

        cma_loss, caption_loss = coca_loss(
            image_features=output_dict["ecg_latent"],
            text_features=output_dict["text_latent"],
            logits=output_dict["logits"],
            labels=output_dict["labels"],
            logit_scale=output_dict["logit_scale"]
        )

        # cma_loss = torch.tensor(0.0).type_as(caption_loss)
        # uot_loss = torch.tensor(0.0).type_as(caption_loss)

        loss_dict = {
            "loss": cma_loss + caption_loss + uot_loss,
            "cma_loss": cma_loss,
            "caption_loss": caption_loss,
            "uot_loss": uot_loss
        }

        if torch.isnan(loss_dict["loss"]):
            ipdb.set_trace()

        # don't write metrics for now
        metrics_dict = {}
        
        return loss_dict, metrics_dict
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        cur_dataset_name = self.val_dataset_list[dataloader_idx]
        class_names = self.dataset_class_names[cur_dataset_name]
        indices = [self.all_labels.index(i) for i in class_names]
        cur_zeroshot_weights = self.zeroshot_weights[:, indices]
        with torch.no_grad():
            ecg_emb = self.encode_ecg(batch['ecg'], normalize=True, proj_contrast=True)
            # if self.local_rank == 0:
            #     print(f"Generated reports in rank {torch.distributed.get_rank()}")
            #     ecg_out = self.generate(batch['ecg'][:4], generation_type="beam_search")
            #     print(self.tokenizer.batch_decode(ecg_out[0]))

        cur_logits = torch.matmul(ecg_emb, cur_zeroshot_weights)
        self.val_step_outputs.append({
            'dataloader_idx': dataloader_idx,
            'logits': cur_logits,
            'label': batch['label']
        })

    def generate(
        self,
        ecg,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,    # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"
        device = ecg.device

        with torch.no_grad():
            sot_token_id = _token_to_tensor(self.tokenizer.convert_tokens_to_ids("[BOS]") if sot_token_id is None else sot_token_id, device=device)
            eos_token_id = _token_to_tensor(self.tokenizer.convert_tokens_to_ids("[SEP]") if eos_token_id is None else eos_token_id, device=device)
            pad_token_id = self.tokenizer.pad_token_type_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]
            stopping_criteria = StoppingCriteriaList(stopping_criteria)

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    ecg_inputs=ecg,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    pad_len = seq_len - output.shape[1]
                    return torch.cat((
                            output,
                            torch.ones(output.shape[0], pad_len, device=device, dtype=output.dtype) * pad_token_id
                        ),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, _, image_embs = self._encode_ecg(ecg)

            if text is None:
                text = torch.ones((ecg.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self.forward_for_logits(
                    ecg,
                    x,
                    ecg_latent=image_latent,
                    ecg_embs=image_embs,
                    output_labels=False,
                )["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if all(stopping_criteria(out, None)):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def _generate_beamsearch(
            self,
            ecg_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=1,
            num_beam_groups=1,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
    ):
        device = ecg_inputs.device
        batch_size = ecg_inputs.shape[0]
        ecg_inputs = torch.repeat_interleave(ecg_inputs, num_beams, dim=0)
        image_latent, _, image_embs = self._encode_ecg(ecg_inputs)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=ecg_inputs)
            outputs = self.forward_for_logits(
                model_inputs['images'],
                model_inputs['text'],
                ecg_latent=image_latent,
                ecg_embs=image_embs,
                output_labels=False,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


if __name__ == "__main__":
    # from melp.datasets.pretrain_datamodule import ECGTextDataModule
    # dm = ECGTextDataModule(
    #     dataset_dir="/disk1/*/ECG/raw",
    #     dataset_list=["mimic-iv-ecg"],
    #     val_dataset_list=None,
    #     batch_size=4,
    #     num_workers=1,
    #     train_data_pct=0.1,
    # )
    
    # for batch in dm.val_dataloader():
    #     break
    
    model = MELPModel(ecg_encoder_name="ecgfm")
    # out = model.shared_step(batch, 0)
    ipdb.set_trace()

