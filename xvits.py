import torch
import torch.nn as nn

from timm.models.layers import to_2tuple, Mlp, DropPath
from vits import AbsolutePositionEmbeds, Attention, CrossPositionAttentionStem, Similarity


class CrossAttention(nn.Module):
    def __init__(self, gs, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.gs = gs
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.similarity = Similarity(head_dim)

    def forward(self, q, kv):
        (B, Lq, C), Lkv = q.shape, kv.shape[1]
        q = self.q_proj(q).reshape(B, Lq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv_proj(kv).reshape(B, Lkv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = self.similarity(q, k)

        with torch.cuda.amp.autocast(enabled=False):
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        q = out.transpose(1, 2).reshape(B, Lq, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q


class CrossBlock(nn.Module):
    def __init__(self, gs, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            gs, dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2q = norm_layer(dim)
        self.norm2kv = norm_layer(dim)
        self.cross_attn = CrossAttention(
            gs, dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv):
        q = q + self.drop_path(self.self_attn(self.norm1(q)))
        q = q + self.drop_path(self.cross_attn(self.norm2q(q), self.norm2kv(kv)))
        q = q + self.drop_path(self.mlp(self.norm3(q)))
        return q


class CrossDecoder(nn.Module):
    def __init__(self, grid_size=(14, 14), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                 embed_layer=None, patchify=None, head=None, norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0., drop_path=0.,avg_vis_mask_token=True, freeze_pe=True):
        super().__init__()

        grid_size = to_2tuple(grid_size)
        self.patchify = patchify
        self.embed = embed_layer
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        self.pos_embed = AbsolutePositionEmbeds(grid_size, embed_dim, freeze=freeze_pe)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.avg_vis_mask_token = avg_vis_mask_token
        if avg_vis_mask_token:
            self.mask_proj = CrossPositionAttentionStem(embed_dim)

        self.blocks = nn.ModuleList([
            CrossBlock(grid_size, embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                       drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                       norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = head

        self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self):
        # initialization
        if self.mask_token is not None:
            torch.nn.init.zeros_(self.mask_token)
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, mean=0, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pid, mask_pid):
        assert x.ndim == 3, 'Input image is not pachified'

        # Embed patches
        x = self.embed(x)
        B, Lx, D = x.shape
        assert Lx == pid.shape[1] + 1

        # Mask tokens
        m = self.mask_token.repeat(B, mask_pid.shape[1], 1)
        if self.avg_vis_mask_token:
            pe_q = self.pos_embed(mask_pid,cls=False) 
            pe_k = self.pos_embed(pid,cls=False)
            m = m + self.mask_proj(pe_q, pe_k, x[:, 1:])
        m = torch.cat((self.cls_token.repeat(B, 1, 1), m), dim=1)

        # Add PEs
        x = x + self.pos_embed(pid)
        m = m + self.pos_embed(mask_pid)

        # apply Transformer blocks
        for blk in self.blocks:
            m = blk(q=m, kv=x)
        m = self.norm(m)

        if self.head is not None:
            m = self.head(m)
        return m


