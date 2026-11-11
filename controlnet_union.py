"""ControlNet-Union, ported from diffusers' ControlNetUnionModel (diffusers/models/controlnets/controlnet_union.py,
v0.35), which is xinsir's ControlNet++ (xinsir6/ControlNetPlus). Our diffusers (0.31) does not have it.

One ControlNet takes several hints. Each hint goes through the same hint encoder; a small transformer over
[hint_1 + task_1, ..., hint_n + task_n, noisy sample] gives each hint a per-channel offset, and the hints are summed
into the sample. Which hints are on is also a 0/1 vector that is embedded like a timestep and added to the time
embedding.

Changes from the reference:
- It is a subclass of our ControlNetModel, so the encoder, hint encoder, zero convolutions and from_unet are the
  ones we already use. Only the union modules are new, with the same names as in the reference.
- The SDXL-only parts (text_time embedding, guess mode, global pooling, MultiControlNet scaling) are left out.
- The reference sizes the control type embedding with SDXL's addition_time_embed_dim (256). SD2 has none, so it is
  control_type_embed_dim here, 256 by default.
- The reference's ResidualAttentionBlock comes from CLIP, which runs nn.MultiheadAttention sequence-first, but the
  reference feeds it batch-first tensors. So its fusion transformer attends across the batch instead of across the
  hints. We use batch_first=True.
"""
import torch
from torch import nn
from diffusers import ControlNetModel
from diffusers.models.controlnet import zero_module
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionMlp(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = ResidualAttentionMlp(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ControlNetUnionModel(ControlNetModel):
    def __init__(
        self,
        num_control_type=2,
        num_trans_channel=320,
        num_trans_head=8,
        num_trans_layer=1,
        num_proj_channel=320,
        control_type_embed_dim=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_to_config(
            num_control_type=num_control_type,
            num_trans_channel=num_trans_channel,
            num_trans_head=num_trans_head,
            num_trans_layer=num_trans_layer,
            num_proj_channel=num_proj_channel,
            control_type_embed_dim=control_type_embed_dim,
        )

        time_embed_dim = self.config.block_out_channels[0] * 4

        task_scale_factor = num_trans_channel**0.5
        self.task_embedding = nn.Parameter(task_scale_factor * torch.randn(num_control_type, num_trans_channel))
        self.transformer_layes = nn.ModuleList(
            [ResidualAttentionBlock(num_trans_channel, num_trans_head) for _ in range(num_trans_layer)]
        )
        self.spatial_ch_projs = zero_module(nn.Linear(num_trans_channel, num_proj_channel))
        self.control_type_proj = Timesteps(control_type_embed_dim, self.config.flip_sin_to_cos, self.config.freq_shift)
        self.control_add_embedding = TimestepEmbedding(control_type_embed_dim * num_control_type, time_embed_dim)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        control_type,
        control_type_idx,
        conditioning_scale=1.0,
    ):
        """
        controlnet_cond: list of hint images, (B, 3, H, W) each, in [0, 1].
        control_type: (B, num_control_type), 1 where that hint is given.
        control_type_idx: the control type of each entry of controlnet_cond.
        conditioning_scale: a float, or one per hint.
        """
        if isinstance(conditioning_scale, float):
            conditioning_scale = [conditioning_scale] * len(controlnet_cond)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        control_embeds = self.control_type_proj(control_type.flatten())
        control_embeds = control_embeds.reshape((t_emb.shape[0], -1)).to(emb.dtype)
        emb = emb + self.control_add_embedding(control_embeds)

        # 2. pre-process: fuse the hints into the sample
        sample = self.conv_in(sample)

        inputs = []
        condition_list = []
        for cond, control_idx, scale in zip(controlnet_cond, control_type_idx, conditioning_scale):
            condition = self.controlnet_cond_embedding(cond)
            feat_seq = torch.mean(condition, dim=(2, 3))
            feat_seq = feat_seq + self.task_embedding[control_idx]
            inputs.append(feat_seq.unsqueeze(1) * scale)
            condition_list.append(condition * scale)

        inputs.append(torch.mean(sample, dim=(2, 3)).unsqueeze(1))

        x = torch.cat(inputs, dim=1)
        for layer in self.transformer_layes:
            x = layer(x)

        controlnet_cond_fuser = sample * 0.0
        for idx, (condition, scale) in enumerate(zip(condition_list, conditioning_scale)):
            alpha = self.spatial_ch_projs(x[:, idx])[:, :, None, None]
            controlnet_cond_fuser = controlnet_cond_fuser + condition + alpha * scale

        sample = sample + controlnet_cond_fuser

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. zero convolutions
        down_block_res_samples = [
            controlnet_block(res_sample)
            for res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks)
        ]
        mid_block_res_sample = self.controlnet_mid_block(sample)

        return down_block_res_samples, mid_block_res_sample
