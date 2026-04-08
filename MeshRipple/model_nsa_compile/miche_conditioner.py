import torch
from torch import nn
from beartype import beartype
from miche.encode import load_model
import os
# helper functions

def exists(val):
    return val is not None

def default(*values):
    for value in values:
        if exists(value):
            return value
    return None


# point-cloud encoder from Michelangelo
@beartype
class PointConditioner(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_latent = None,
        model_name = 'miche-256-feature',
        cond_dim = 768,
        feature_dim = 768,
        freeze = True,
    ):
        super().__init__()

        # open-source version of miche
        if model_name == 'miche-256-feature':
            # Resolve paths dynamically
            current_dir = os.path.dirname(os.path.realpath(__file__))
            miche_root = os.path.join(os.path.dirname(current_dir), "miche")
            
            # Check for ComfyUI models directory (go up 3 levels from MeshRipple/model_nsa_compile to custom_nodes, then 1 more to root)
            comfy_models_miche = os.path.abspath(os.path.join(current_dir, "../../../../../models/miche/"))
            
            ckpt_name = "shapevae-256.ckpt"
            yaml_name = "shapevae-256.yaml"
            
            # Prioritize ComfyUI/models/miche/
            primary_ckpt = os.path.join(comfy_models_miche, ckpt_name)
            primary_yaml = os.path.join(comfy_models_miche, yaml_name)
            
            if os.path.exists(primary_ckpt):
                ckpt_path = primary_ckpt
                config_path = primary_yaml
            else:
                # Fallback to local extension dir
                ckpt_path = os.path.join(miche_root, ckpt_name)
                config_path = os.path.join(miche_root, yaml_name)
            
            if not (ckpt_path and os.path.exists(ckpt_path)):
                ckpt_path=None
                print(f'[WARNING] Michelangelo ckpt not found at {primary_ckpt} or {os.path.join(miche_root, ckpt_name)}')

            self.feature_dim = feature_dim    # embedding dimension
            self.cond_length = 257     # length of embedding
            self.point_encoder = load_model(ckpt_path=ckpt_path, config_path=config_path)
            
            # additional layers to connect miche and GPT
            self.cond_head_proj = nn.Linear(cond_dim, self.feature_dim)
            self.cond_proj = nn.Linear(cond_dim, self.feature_dim)
            
        else:
            raise NotImplementedError

        # whether to finetuen point-cloud encoder
        if freeze:
            for parameter in self.point_encoder.parameters():
                parameter.requires_grad = False

        self.freeze = freeze
        self.model_name = model_name
        self.dim_latent = self.feature_dim
        
        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)


    @property
    def device(self):
        return next(self.buffers()).device


    def embed_pc(self, pc_normal):
        # encode point cloud to embeddings
        if self.model_name == 'miche-256-feature':
            point_feature = self.point_encoder.encode_latents(pc_normal)
            pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
            pc_embed = self.cond_proj(point_feature[:, 1:])
            pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)

        return pc_embed


    def forward(
        self,
        pc = None,
        pc_embeds = None,
    ):
        if pc_embeds is None:
            pc_embeds = self.embed_pc(pc.to(next(self.buffers()).dtype))
            
        assert not torch.any(torch.isnan(pc_embeds)), 'NAN values in pc embedings'
        
        return pc_embeds
    
