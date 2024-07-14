from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from vits import VIT, Patchify, PatchEmbed
from xvits import CrossDecoder


def _build_mlp(in_dim, out_dim, hidden_dim=4096, num_layers=1, norm=nn.LayerNorm, out_norm=False):
    if num_layers == 0:
        return None
    projector = nn.Sequential()
    for l in range(num_layers):
        dim1 = in_dim if l == 0 else hidden_dim
        dim2 = out_dim if l == num_layers - 1 else hidden_dim
        projector.add_module(f"linear{l}", nn.Linear(dim1, dim2, bias=False))
        if out_norm or l < num_layers - 1:
            projector.add_module(f"norm{l}", norm(dim2))
        if l < num_layers - 1:
            projector.add_module(f"act{l}", nn.GELU())
    return projector


class LMIM(nn.Module):
    """ Latent Masked Image Modeling with VisionTransformer backbone
    """
    def __init__(self, grid_size=14, patch_size=16, patch_gap=0, in_chans=3,
                 embed_dim=1024, num_heads=16, depth=24, target_depth=0, 
                 decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0., drop_path=0.,
                 tau=0.2, num_vis=20, avg_vis_mask_token=True,
                 avg_sim_coeff=0., mask_target=True,
                 loss='infonce_patches', freeze_pe=None, proj_cfg=None):
        super().__init__()

        self.loss = loss
        self.patch_gap = patch_gap
        self.tau = tau
        self.num_vis = num_vis
        self.avg_sim_coeff = avg_sim_coeff
        
        self.patchify = Patchify(patch_size=patch_size, grid_size=grid_size)
        self.mask_target = mask_target

        # --------------------------------------------------------------------------
        # Encoder
        embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        head = _build_mlp(embed_dim, embed_dim, proj_cfg.mlp_dim, proj_cfg.mlp_depth, out_norm=True)
        self.encoder = VIT(patchify=self.patchify, embed_layer=embed,
                           grid_size=grid_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, mask=False, cls=True,
                           drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)
        # --------------------------------------------------------------------------
        # Target Encoder
        embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        head = _build_mlp(embed_dim, embed_dim, proj_cfg.mlp_dim, proj_cfg.mlp_depth, out_norm=True)
        self.target_encoder = VIT(patchify=self.patchify, embed_layer=embed,
                                  grid_size=grid_size, embed_dim=embed_dim, num_heads=num_heads, depth=target_depth,
                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, mask=False, cls=True,
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)

        # --------------------------------------------------------------------------
        # Feature Decoder and Predictor
        embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        head = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to patch
        self.decoder = CrossDecoder(embed_layer=embed, grid_size=grid_size,
                                    embed_dim=decoder_embed_dim, num_heads=decoder_num_heads, depth=decoder_depth,
                                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, avg_vis_mask_token=avg_vis_mask_token,
                                    drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)

        self._init_target_encoder()

    @torch.no_grad()
    def _init_target_encoder(self):
        """Initialize Momentum Encoder"""
        for p, q_ema in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            q_ema.data[:] = p.data[:]
            q_ema.requires_grad = False


    @torch.no_grad()
    def _update_target_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for p, q_ema in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            q_ema.data = q_ema.data * m + p.data * (1. - m)

    @torch.no_grad()
    def create_views(self, patch_pix):
        N, L, D = patch_pix.shape
        pid = torch.arange(L, device=patch_pix.device).long()

        # sample visible and target patches
        shuffle = torch.argsort(torch.rand(N, L, device=patch_pix.device), dim=1)       # ascend: small is keep, large is remove
        vis_idx = pid[shuffle[:, :self.num_vis]]      # idx of vis patches
        mask_idx = pid[shuffle[:, self.num_vis:]]
        if self.mask_target:
            trg_idx = mask_idx
        else:
            trg_idx = pid[shuffle]

        # gather patches
        vis_pix = patch_pix.gather(dim=1, index=vis_idx[:, :, None].repeat(1, 1, D))
        trg_pix = patch_pix.gather(dim=1, index=trg_idx[:, :, None].repeat(1, 1, D))
        return vis_pix, vis_idx, trg_pix, trg_idx, mask_idx

    def forward_loss(self, pred, target, pred2unit=True):
        if self.loss == 'norm_l2':
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            loss = F.mse_loss(pred, target)

        elif self.loss == 'infonce_patches':
            bs, nt, d = pred.shape
            
            if pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            scores = torch.einsum('npd,nqd->npq', pred, target) / self.tau
            labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(bs, 1)
            loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1)) * (self.tau * 2)

        else:
            raise NotImplementedError(f"Loss type {self.loss} not implemented")
        return loss

    def forward(self, imgs, mom=0.99, sim_trg=0.75,  update_ema=False):
        # Image to patches
        patch_pix = self.patchify(imgs, self.patch_gap)

        # Create two views by masking
        pix_vis, pid_vis, pix_trg, pid_trg, mask_idx = self.create_views(patch_pix)

        # Compute targets
        with torch.no_grad():
            if update_ema:
                self._update_target_encoder(mom)  # update the momentum encoder
            Lm = mask_idx.shape[-1]
            x_trg = self.target_encoder(pix_trg, pid_trg)[:, -Lm:]

        # Forward encoder
        x_vis = self.encoder(pix_vis, pid_vis)

        # Forward decoder
        x_pred = self.decoder(x_vis, pid_vis, mask_pid=mask_idx)[:, 1:]

        # Compute loss
        loss = self.forward_loss(x_pred, x_trg)
        if self.avg_sim_coeff > 0:
            loss += self.avg_sim_coeff * (avg_pairwise_sim(x_pred, x_pred)-sim_trg).pow(2)
            loss += self.avg_sim_coeff * (avg_pairwise_sim(x_vis[:, 1:], x_vis[:, 1:])-sim_trg).pow(2)
            
        # Log metrics
        metrics = {
            'avg_sim_pred': avg_pairwise_sim(x_pred, x_pred).item(),
            'avg_sim_vis': avg_pairwise_sim(x_vis[:, 1:], x_vis[:, 1:]).item(),
            'avg_sim_trg': avg_pairwise_sim(x_trg, x_trg).item(),
        }
        return loss, metrics


def avg_pairwise_sim(q,k):
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    if len(q.shape) == 3 and len(k.shape) == 3:
        return torch.einsum('npd,nqd->npq', q, k).mean()
    else:
        return torch.einsum('nhpd,nhqd->nhpq', q, k).mean()


CFG = {
    'vit_tiny':   {'embed_dim': 192,  'depth': 12, 'num_heads': 3, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_small':  {'embed_dim': 384,  'depth': 12, 'num_heads': 6, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_base':   {'embed_dim': 768,  'depth': 12, 'num_heads': 12, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_large':  {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_huge':   {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 14},
}

def build_lmim(backbone, decoder_depth=3, decoder_embed_dim=512, decoder_num_heads=16, **kwargs):
    cfg = CFG[backbone]
    model = LMIM(
        patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'], depth=cfg['depth'], num_heads=cfg['num_heads'],
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=cfg['mlp_ratio'], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
