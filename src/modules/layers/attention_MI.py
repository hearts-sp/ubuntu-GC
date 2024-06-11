import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EntityAttentionLayer_MI(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, args):
        super(EntityAttentionLayer_MI, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_heads = args.attn_n_heads
        self.n_agents = args.n_agents
        self.args = args

        assert self.embed_dim % self.n_heads == 0, "Embed dim must be divisible by n_heads"
        self.head_dim = self.embed_dim // self.n_heads
        self.register_buffer('scale_factor',
                             th.scalar_tensor(self.head_dim).sqrt())

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim * 3, bias=False)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, entities, pre_mask=None, post_mask=None, ret_attn_logits=False):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before attention.
            shape: batch_size, # of agents, # of entities
        post_mask: Which agents/entities are not available. Zero out their outputs to
                   prevent gradients from flowing back. Shape of 2nd dim determines
                   whether to compute queries for all entities or just agents.
            shape: batch size, # of agents (or entities)

        Return shape: batch size, # of agents, embedding dimension
        """
        entities_t = entities.transpose(0, 1)
        n_queries = post_mask.shape[1] # bs, na
        pre_mask = pre_mask[:, :n_queries] # bs, na, ne
        ne, bs, ed = entities_t.shape
        query, key, value = self.in_trans(entities_t).chunk(3, dim=2)

        query = query[:n_queries]

        query_spl = query.reshape(n_queries, bs * self.n_heads, self.head_dim).transpose(0, 1) # hs * bs, na, h_dim
        key_spl = key.reshape(ne, bs * self.n_heads, self.head_dim).permute(1, 2, 0) # hs * bs, h_dim, ne
        value_spl = value.reshape(ne, bs * self.n_heads, self.head_dim).transpose(0, 1) # hs * bs, ne, h_dim

        attn_logits = th.bmm(query_spl, key_spl) / self.scale_factor # hs * bs, na, ne
        if pre_mask is not None:
            pre_mask_rep = pre_mask.repeat_interleave(self.n_heads, dim=0) # hs * bs, na, ne
            masked_attn_logits = attn_logits.masked_fill(pre_mask_rep[:, :, :ne].bool(), -float('Inf'))
        attn_weights = F.softmax(masked_attn_logits, dim=2)
        # some weights might be NaN (if agent is inactive and all entities were masked)
        attn_weights = attn_weights.masked_fill((attn_weights != attn_weights).bool(), 0) # hs * bs, na, ne
        attn_outs = th.bmm(attn_weights, value_spl) # hs * bs, na, h_dim
        attn_outs = attn_outs.transpose(
            0, 1).reshape(n_queries, bs, self.embed_dim) # na, bs, hs * d_dim
        attn_outs = attn_outs.transpose(0, 1) # bs, na, dim
        attn_outs = self.out_trans(attn_outs)
        if post_mask is not None:
            attn_outs = attn_outs.masked_fill(post_mask.bool().unsqueeze(2), 0)

        if ret_attn_logits:
            return attn_outs, attn_logits.view(bs, self.n_heads, n_queries, ne)

        return attn_outs

