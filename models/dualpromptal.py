'''
TODO: E-prompt (prompt pool 크기와 e-prompt 개수를 몇으로? 원래는 task 수 만큼 생성하지만 AL에서는 class만큼? 하지만 그러기엔 class가 많은 경우 비효율적임)
TODO: Prefix-tuning
TODO: Matching loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from timm.models.vision_transformer import Block
    
class PrefixAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        # attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PrefixBlock(Block):

    def __init__(self, **kwargs):
        super(PrefixBlock, self).__init__(**kwargs)
        self.attn = PrefixAttention(
            dim       = kwargs['dim'],
            num_heads = kwargs['num_heads'],
            qkv_bias  = kwargs['qkv_bias'],
            attn_drop = kwargs['attn_drop'],
            proj_drop = kwargs['proj_drop'],
        )
        
    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt=prompt)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, pool_size=10, num_layers=1, num_heads=-1):
        super().__init__()

        self.length = length
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        assert embed_dim % self.num_heads == 0
        
        # E-prompt pool
        prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, self.num_heads, embed_dim // self.num_heads)
        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) 
        nn.init.uniform_(self.prompt, -1, 1)

        # prompt classifier
        self.prompt_clf = nn.Linear(embed_dim, pool_size)
    
    def forward(self, x_embed, cls_features=None):
        out = {}
    
        # inputs embedding key
        if cls_features is None:
            x_embed = torch.max(x_embed, dim=1)[0] # batch_size, dim
        else:
            x_embed = cls_features
        
        prompt_logits = self.prompt_clf(x_embed)
        out['prompt_logits'] = prompt_logits

        selected_key_idx = prompt_logits.argmax(dim=1)
        e_prompt = self.prompt[:,:,selected_key_idx] # num_layers, kv, batch_size, length, num_heads, heads_embed_dim
        e_prompt = e_prompt.permute(0,2,1,3,4,5) # num_layers, batch_size, kv, length, num_heads, heads_embed_dim
        out['e_prompt'] = e_prompt

        return out
    
    
class DualPromptAL(nn.Module):
    def __init__(
        self, encoder_name: str, num_classes: int, img_size: int, pretrained: bool, 
        g_prompt_layer_idx: list = [0, 1], g_prompt_length: int = 5, e_prompt_layer_idx: list = [2, 3, 4], e_prompt_length: int = 5):
        super(DualPromptAL, self).__init__()
        
        self.encoder = create_model(
            model_name  = encoder_name, 
            num_classes = num_classes, 
            img_size    = img_size, 
            pretrained  = pretrained,
            block_fn    = PrefixBlock
        )
        for n, p in self.encoder.named_parameters():
            if 'head' not in n:
                p.required_grad = False
        
        self.num_heads = self.encoder.blocks[0].attn.num_heads

        # G-prompt for invariant features
        self.g_prompt_layer_idx = g_prompt_layer_idx
        num_g_prompt = len(self.g_prompt_layer_idx) 
        
        g_prompt_shape = (num_g_prompt, 2, g_prompt_length, self.num_heads, self.encoder.embed_dim // self.num_heads)
        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
        nn.init.uniform_(self.g_prompt, -1, 1)
        
        # E-prompt for class features
        self.e_prompt_layer_idx = e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx)
        
        self.e_prompt = EPrompt(
            length     = e_prompt_length, 
            embed_dim  = self.encoder.embed_dim, 
            pool_size  = num_classes, 
            num_layers = num_e_prompt, 
            num_heads  = self.num_heads
        )
    
    
    def train(self, mode=True):
        if mode:
            # train
            self.encoder.eval()
        else:
            # eval
            for module in self.children():
                module.train(mode)
        

    def forward_features(self, x, cls_features=None):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)
        
        g_prompt_counter = -1
        e_prompt_counter = -1

        out = self.e_prompt(x, cls_features=cls_features)
        e_prompt = out['e_prompt']

        for i, block in enumerate(self.encoder.blocks):
            if i in self.g_prompt_layer_idx:
                
                g_prompt_counter += 1
                # Prefix tunning, [B, 2, g_prompt_length, num_heads, embed_dim // num_heads]
                idx = torch.tensor([g_prompt_counter] * x.shape[0]).to(x.device)
                g_prompt = self.g_prompt[idx]
                
                x = block(x, prompt=g_prompt)
            
            elif i in self.e_prompt_layer_idx:
                e_prompt_counter += 1
                
                # Prefix tunning, [B, 2, e_prompt_length, num_heads, embed_dim // num_heads]
                x = block(x, prompt=e_prompt[e_prompt_counter])
                
            else:
                x = block(x)

        x = self.encoder.norm(x)
        out['embeddings'] = x

        return out

    def forward_head(self, out):
        # pre-logits
        x = out['embeddings'][:, 0] # CLS embedding
        x = self.encoder.fc_norm(x)
        x = self.encoder.head_drop(x)
        
        # logits
        out['logits'] = self.encoder.head(x)
        
        return out

    def forward(self, x, cls_features=None):
        out = self.forward_features(x, cls_features=cls_features)
        out = self.forward_head(out)
        return out
