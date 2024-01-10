from timm import create_model
from functools import reduce
from operator import mul
import math
import torch
import torch.nn as nn


class Prompt(nn.Module):
    def __init__(self, nb_blocks: int, embed_dim: int, patch_size: int, prompt_tokens: int = 5, prompt_dropout: float = 0.0, prompt_type: str = 'shallow'):
        super(Prompt, self).__init__()
        
        # prompt
        self.prompt_tokens = prompt_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.prompt_dim = embed_dim
        self.prompt_type = prompt_type # "shallow" or "deep"
        assert self.prompt_type in ['shallow','deep'], "prompt type should be 'shallow' or 'deep'."

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_tokens, self.prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.prompt_type == 'deep':  # noqa
            self.total_d_layer = nb_blocks
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(self.total_d_layer-1, self.prompt_tokens, self.prompt_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)


class VPTAL(nn.Module):
    def __init__(
        self, encoder_name: str, num_classes: int, img_size: int, patch_size: int, pretrained: bool = True, head_type: str = 'prompt',
        prompt_tokens: int = 5, prompt_dropout: float = 0.0, prompt_type: str = 'shallow'
    ):
        super(VPTAL, self).__init__()
                
        self.encoder = create_model(
            model_name  = encoder_name, 
            num_classes = num_classes, 
            pretrained  = pretrained, 
            img_size    = img_size,
            patch_size  = patch_size
        )
        
        self.head_type = head_type
            
        # freeze
        for n, p in self.encoder.named_parameters():
            if 'head' not in n:
                p.requires_grad = False
                
        # prompt
        self.prompt = Prompt(
            nb_blocks = len(self.encoder.blocks),
            embed_dim = self.encoder.embed_dim,
            patch_size = self.encoder.patch_embed.patch_size,
            prompt_tokens = prompt_tokens,
            prompt_dropout = prompt_dropout,
            prompt_type = prompt_type
        )
        self.prompt.head = self.encoder.head
            
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.prompt.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
                        
    def incorporate_prompt(self, x, prompt_embeddings, n_prompt: int = 0):
        B = x.shape[0]
        
        # concat prompts: (batch size, cls_token + n_prompt + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
            x[:, (1+n_prompt):, :]
        ), dim=1)
        
        return x
    
    def forward_features(self, x):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)
        
        # add prompts
        x = self.incorporate_prompt(x, prompt_embeddings=self.prompt.prompt_embeddings)
        
        if self.prompt.prompt_type == 'deep':
            # deep mode
            x = self.encoder.blocks[0](x)
            for i in range(1, self.prompt.total_d_layer):
                x = self.incorporate_prompt(x, prompt_embeddings=self.prompt.deep_prompt_embeddings[i-1], n_prompt=self.prompt.prompt_tokens)
                x = self.encoder.blocks[i](x)
        else:
            # shallow mode
            x = self.encoder.blocks(x)
            
        x = self.encoder.norm(x)
        return x
    
    def forward_head(self, x):
        # pre-logits
        if self.head_type == 'prompt':
            x = x[:, 1:1+self.prompt.prompt_tokens].mean(dim=1)
        elif self.head_type == 'cls':
            x = x[:, 0] # CLS embedding
            
        x = self.encoder.fc_norm(x)
        x = self.encoder.head_drop(x)
        
        # logits
        x = self.prompt.head(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x