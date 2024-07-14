import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.layers import to_2tuple, Mlp, DropPath

class CrossPositionAttentionStem(nn.Module):
    """ Positional-weighted Vision Cues as Part of the Mask Token"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, pe_q, pe_k, kv):
        # Compute position weighted attention
        attn = pe_q @ pe_k.transpose(-2, -1) * self.scale 
        attn = attn.softmax(dim=-1)
        return attn @ kv

class PatchEmbed(nn.Module):
    """ Linear Patch Embedding"""
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Linear(in_chans*patch_size[0]*patch_size[1], embed_dim)

    def forward(self, x):
        return self.proj(x)


class Patchify:
    """ Image to Patch"""
    def __init__(self, patch_size, grid_size=None):
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = to_2tuple(grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def patchify(self, imgs, gap):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        bs = imgs.shape[0]
        ph, pw = self.patch_size
        h, w = imgs.shape[-2:]
        assert w % (pw + gap) == 0 and h % (ph + gap) == 0

        th, tw = h // (ph + gap), w // (pw + gap)
        x = imgs.reshape(shape=(bs, 3, th, ph+gap, tw, pw+gap))
        x = torch.einsum('nchpwq->nhwpqc', x)
        if gap > 0:
            stx = (torch.randint(0, gap, x.shape[:3]).unsqueeze(-1) + torch.arange(ph)).to(x.device)
            sty = (torch.randint(0, gap, x.shape[:3]).unsqueeze(-1) + torch.arange(pw)).to(x.device)
            x = x.gather(dim=3, index=stx[:, :, :, :, None, None].repeat(1, 1, 1, 1, x.shape[4], x.shape[5]))
            x = x.gather(dim=4, index=sty[:, :, :, None, :, None].repeat(1, 1, 1, x.shape[3], 1, x.shape[5]))
        x = x.reshape(shape=(bs, th * tw, ph * pw * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        bs = x.shape[0]
        ph, pw = self.patch_size
        gh, gw = self.grid_size
        assert gh * gw == x.shape[1]

        x = x.reshape(shape=(bs, gh, gw, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(bs, 3, gh * ph, gw * pw))
        return imgs

    def __call__(self, x, gap):
        return self.patchify(x, gap)


class SinCosPEs(nn.Module):
    """ SinCos Positional Embeddings"""
    def __init__(self, embed_dim, input_rng=(0., 1.), target_range=(0., 14.)):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_range = input_rng
        self.target_range = target_range
        assert embed_dim % 4 == 0

    def sincos_embedding(self, pos):
        """
        pos: a list of positions to be encoded: size (B, L)
        out: (M, D)
        """
        omega = torch.arange(self.embed_dim // 4, device=pos.device) / self.embed_dim * 4.
        omega = 1. / 10000**omega  # (D/4,)

        if pos.ndim == 3:
            pos = pos.squeeze(2)    # (B, L)
        scale = (self.target_range[1] - self.target_range[0]) / (self.input_range[1] - self.input_range[0])
        pos_scaled = (pos - self.input_range[0]) * scale + self.target_range[0]

        out = torch.einsum('ml,d->mld', pos_scaled, omega)  # (B, L, D/4), outer product
        emb = torch.cat((torch.sin(out), torch.cos(out)), dim=-1)
        return emb

    def forward(self, loc):
        y, x = loc.chunk(2, dim=-1)
        x_pe = self.sincos_embedding(x)
        y_pe = self.sincos_embedding(y)
        pe = torch.cat((x_pe, y_pe), dim=-1)
        return pe


class AbsolutePositionEmbeds(nn.Module):
    def __init__(self, grid_size, dim, freeze=True):
        super().__init__()
        self.grid_size = grid_size

        gY, gX = grid_size
        L = gX * gY + 1
        if freeze:
            self.register_buffer('pos_embeds', torch.zeros(L, dim))
        else:
            self.pos_embeds = nn.Parameter(torch.zeros(L, dim))

        # Initialization
        loc = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gX),
            torch.linspace(0, 1, gY)
        ), 2).flatten(0, 1)
        pes = SinCosPEs(dim, target_range=(0, grid_size[0]))(loc).squeeze(1)

        self.pos_embeds.data[1:, ] = pes

    def forward(self, pid, cls=True):
        pid = pid + 1 # pid doesn't include cls token
        if cls:
            pid = F.pad(pid, (1, 0, 0, 0), value=0)
        return self.pos_embeds[pid]

class Similarity(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, q, k):
        return q @ k.transpose(-2, -1) * self.scale


class Attention(nn.Module):
    def __init__(self, gs, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.gs = gs
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.similarity = Similarity(head_dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = self.similarity(q, k)

        with torch.cuda.amp.autocast(enabled=False):
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, gs, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            gs, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VIT(nn.Module):
    def __init__(self, grid_size=(14, 14), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                 embed_layer=None, patchify=None, head=None, norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0., drop_path=0., cls=True, mask=False, freeze_pe=True,
                 avg_vis_mask_token=True,):
        super().__init__()

        grid_size = to_2tuple(grid_size)
        self.patchify = patchify
        self.embed = embed_layer
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        
        
        self.pos_embed = AbsolutePositionEmbeds(grid_size, embed_dim, freeze=freeze_pe)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if mask else None
        self.avg_vis_mask_token = avg_vis_mask_token
        if avg_vis_mask_token:
            self.mask_proj = CrossPositionAttentionStem(embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(grid_size, embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                  norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = head

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=.02)
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pid=None, mask_pid=None):
        if x.ndim == 4: # Full images. Patchify first
            x = self.patchify(x, gap=0)
            pid = torch.arange(0, x.shape[1], device=x.device).long()[None].repeat(x.shape[0], 1)

        # Embed patches
        x = self.embed(x)

        # Class token
        B, Lx, D = x.shape
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.repeat(B, 1, 1), x), dim=1)
        assert x.shape[1] == pid.shape[1] + 1

        # Mask tokens
        if mask_pid is not None:
            m = self.mask_token.repeat(B, mask_pid.shape[1], 1)
            if self.avg_vis_mask_token:
                pe_q = self.pos_embed(mask_pid,cls=False) # position only
                pe_k = self.pos_embed(pid,cls=False)
                m = m + self.mask_proj(pe_q, pe_k, x[:, 1:])
            x = torch.cat((x, m), dim=1)
            pid = torch.cat((pid, mask_pid), dim=1)

        # Absolute positional embeddings
        x = x + self.pos_embed(pid)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.head is not None:
            x = self.head(x)
        return x


CFG = {
    'vit_tiny':   {'embed_dim': 192,  'depth': 12, 'num_heads': 3, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_small':  {'embed_dim': 384,  'depth': 12, 'num_heads': 6, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_base':   {'embed_dim': 768,  'depth': 12, 'num_heads': 12, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_large':  {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_huge':   {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 14},
}


def build_vit(backbone, img_size=224, patch_gap=0, in_chans=3, out_chans=None, **kwargs):
    cfg = CFG[backbone]
    grid_size = img_size // (cfg['patch_size'] + patch_gap)

    patchify = Patchify(patch_size=cfg['patch_size'], img_size=img_size, patch_gap=patch_gap)
    embed = PatchEmbed(patch_size=cfg['patch_size'], in_chans=in_chans, embed_dim=cfg['embed_dim'])
    head = nn.Linear(cfg['embed_dim'], out_chans, bias=True) if out_chans is not None else None
    model = VIT(patchify=patchify, embed_layer=embed, head=head,
                grid_size=grid_size, embed_dim=cfg['embed_dim'], num_heads=cfg['num_heads'], depth=cfg['depth'],
                mlp_ratio=cfg['mlp_ratio'], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                mask=False, cls=True, **kwargs)
    return model