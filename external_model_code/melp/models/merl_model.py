import torch
import ipdb
import yaml
import math
import numpy as np
from typing import List
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, f1_score
from melp.backbone.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from melp.backbone.vit1d import vit_tiny, vit_small, vit_middle, vit_base
from melp.backbone.pooling import AttentionPool2d
from melp.models.base_pretrain_model import BasePretrainModel
# from melp.utils.utils_loss import clip_loss
from melp.utils.openclip_loss import ClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH


class MERLModel(BasePretrainModel):
    def __init__(self, 
                 ecg_encoder_name: str = "resnet18",
                 text_encoder_name: str = "ncbi/MedCPT-Query-Encoder",
                 val_dataset_list: List = ["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
                                           "icbeb", "chapman"],
                 shared_emb_dim: int = 256,
                 num_leads: int = 12,
                 num_freeze_layers: int = 6,
                 init_logit_scale: float = np.log(1 / 0.07),
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 *args,
                 **kwargs):
        
        self.num_freeze_layers = num_freeze_layers
        super().__init__(ecg_encoder_name=ecg_encoder_name,
                         text_encoder_name=text_encoder_name,
                         shared_emb_dim=shared_emb_dim,
                         num_leads=num_leads,
                         lr=lr,
                         weight_decay=weight_decay,
                         *args,
                         **kwargs)
        self.save_hyperparameters()
        self.proj_out = shared_emb_dim
        self.proj_hidden = 256
        self.val_dataset_list = val_dataset_list
        self.init_ecg_encoder()
        self.init_text_encoder()

        lshape = []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
    
        with open(PROMPT_PATH, 'r') as f:
            self.prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.all_labels = list(self.prompt_dict.keys())

        with open(DATASET_LABELS_PATH, 'r') as f:
            self.dataset_class_names = yaml.load(f, Loader=yaml.FullLoader)

    def init_ecg_encoder(self):
        if 'resnet' in self.ecg_encoder_name:
            if self.ecg_encoder_name == 'resnet18':
                model = ResNet18()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                     embed_dim=self.proj_out, 
                                                     num_heads=4, 
                                                     output_dim=self.proj_out)
            elif self.ecg_encoder_name == 'resnet34':
                model = ResNet34()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_encoder_name == 'resnet50':
                model = ResNet50()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_encoder_name == 'resnet101':
                model = ResNet101()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)

            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

        if 'vit' in self.ecg_encoder_name:
            if self.ecg_encoder_name == 'vit_tiny':
                model = vit_tiny(num_leads=self.num_leads)
            elif self.ecg_encoder_name == 'vit_small':
                model = vit_small(num_leads=self.num_leads)
            elif self.ecg_encoder_name == 'vit_middle':
                model = vit_middle(num_leads=self.num_leads)
            elif self.ecg_encoder_name == 'vit_base':
                model = vit_base(num_leads=self.num_leads)
            self.proj_e_input = model.width    
            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_e_input, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)

        self.ecg_encoder = model
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def init_text_encoder(self):
        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            self.lm_model = AutoModel.from_pretrained(
                self.text_encoder_name)

            # freeze layers
            for layer_idx in range(self.num_freeze_layers):
                for param in list(self.lm_model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False

            text_encoder_hidden_dim = 768
        elif self.text_encoder_name == "google/flan-t5-small":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True)
            text_encoder_hidden_dim = 512
        elif self.text_encoder_name == "google/flan-t5-base":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True)
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
    
    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=128,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output
    
    def encode_ecg(self, ecg):
        # (bz, 12, 5000) -> (bz, 512, 313)
        ecg_emb = self.ecg_encoder(ecg)
        if 'resnet' in self.ecg_encoder_name:
            # attention pooling (only for resnet models)
            # (bz, 512, 313) -> (bz, 256, 313)
            ecg_emb = self.downconv(ecg_emb)
            # (bz, 256, 313) -> (bz, 1, 256)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            # (bz, 1, 256) -> (bz, 256)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

            ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
        
        if 'vit' in self.ecg_encoder_name:
            proj_ecg_emb = self.proj_e(ecg_emb)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)

        return {
            'proj_ecg_emb': proj_ecg_emb,
            'ecg_emb1': ecg_emb1,
            'ecg_emb2': ecg_emb2
        }

    def encode_text(self, input_ids, attention_mask):
        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            text_emb = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
            sequence_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).type_as(attention_mask).bool()
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            batch_size, _, hidden_size = sequence_output.shape
            text_emb = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
            
        proj_text_emb = self.proj_t(text_emb)
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)

        return {
            'proj_text_emb': proj_text_emb,
            'text_emb': text_emb
        }

    @torch.no_grad()
    def ext_ecg_emb(self, ecg):
        ''' For inference only'''
        if 'resnet' in self.ecg_encoder_name:
            ecg_emb = self.ecg_encoder(ecg)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, att_map = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        if 'vit' in self.ecg_encoder_name:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        return proj_ecg_emb
    
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        ''' For inference only'''
        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            # pooler_output
            text_emb = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
            sequence_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).type_as(attention_mask).bool()
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            batch_size, _, hidden_size = sequence_output.shape
            text_emb = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        else:
            raise NotImplementedError
        text_emb = self.proj_t(text_emb)

        return text_emb

    @torch.no_grad()
    def get_class_emd(self, class_name):
        zeroshot_weights = []
        # compute embedding through model for each class
        for texts in class_name:
            texts = texts.lower()
            texts = [texts] # convert to list
            texts = self._tokenize(texts) # tokenize
            # class_embeddings = self.get_text_emb(texts.input_ids.type_as(self.logit_scale).long(),
            #                                      texts.attention_mask.type_as(self.logit_scale).long(),
            #                                     ) # embed with text encoder
            class_embeddings = self.encode_text(texts.input_ids.type_as(self.logit_scale).long(),
                                                texts.attention_mask.type_as(self.logit_scale).long(),
                                               )

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

        return zeroshot_weights
    
    def shared_step(self, batch, batch_idx):
        # only used in training_step now
        ecg_output = self.encode_ecg(batch['ecg'])

        tokenized_input = self._tokenize(batch['report'])
        input_ids = tokenized_input['input_ids'].type_as(batch['ecg']).long()
        attention_mask = tokenized_input['attention_mask'].type_as(batch['ecg']).long()
        text_output = self.encode_text(input_ids, attention_mask)

        # write infonce loss for lightning 
        loss = ClipLoss(
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            use_horovod=False
        )

        cma_loss = loss(
            ecg_output['proj_ecg_emb'], text_output['proj_text_emb'], self.logit_scale.exp())
        uma_loss = loss(
            ecg_output['ecg_emb1'], ecg_output['ecg_emb2'], torch.tensor(1 / 0.07))

        loss_dict = {
            'loss': cma_loss + uma_loss,
            'cma_loss': cma_loss,
            'uma_loss': uma_loss
        }

        # don't write metrics for now
        metrics_dict = {}
        # metrics_dict = {
        #     'acc1': acc1,
        #     'acc5': acc5
        # }

        return loss_dict, metrics_dict

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100)) 

    def on_validation_epoch_start(self, *args, **kargs):
        val_prompts = [self.prompt_dict[i] for i in self.all_labels]
        self.zeroshot_weights = self.get_class_emd(val_prompts)
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx):
        cur_dataset_name = self.val_dataset_list[dataloader_idx]
        class_names = self.dataset_class_names[cur_dataset_name]
        indices = [self.all_labels.index(i) for i in class_names]
        cur_zeroshot_weights = self.zeroshot_weights[:, indices]
        ecg_emb = self.ext_ecg_emb(batch['ecg'])
        ecg_emb /= ecg_emb.norm(dim=-1, keepdim=True)
        cur_logits = torch.matmul(ecg_emb, cur_zeroshot_weights)
        self.val_step_outputs.append({
            'dataloader_idx': dataloader_idx,
            'logits': cur_logits,
            'label': batch['label']
        })

    def on_validation_epoch_end(self, *args, **kargs):
        logits_dict = defaultdict(list)
        labels_dict = defaultdict(list)
        for output in self.val_step_outputs:
            dataloader_idx = output['dataloader_idx']
            logits = output['logits']
            labels = output['label']
            logits_dict[dataloader_idx].append(logits)
            labels_dict[dataloader_idx].append(labels)

        # for each dataset
        dataset_aurocs = []
        for k in logits_dict.keys():
            logits = torch.cat(logits_dict[k], dim=0).float().cpu().numpy()
            labels = torch.cat(labels_dict[k], dim=0).float().cpu().numpy()

            assert logits.shape[1] == labels.shape[1], "Number of classes mismatch"
            
            num_labels = logits.shape[1]
            AUROCs = []
            for i in range(num_labels):
                if len(np.unique(labels[:, i])) == 1:
                    continue
                AUROCs.append(roc_auc_score(labels[:, i], logits[:, i], average='macro', multi_class='ovo'))
            dataset_name = self.val_dataset_list[k]
            self.log(f'val/{dataset_name}_AUROC', np.mean(AUROCs), on_epoch=True, prog_bar=False, sync_dist=True)
            dataset_aurocs.append(np.mean(AUROCs))

        self.log(f'val/mean_AUROC', np.mean(dataset_aurocs), on_epoch=True, prog_bar=True, sync_dist=True)


