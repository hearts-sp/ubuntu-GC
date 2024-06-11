import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer_MI

class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        self.attn = EntityAttentionLayer_MI(hypernet_embed,
                                        hypernet_embed,
                                        hypernet_embed, args)

        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None):
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask)
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.bool().unsqueeze(2), 0)
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            return x3.mean(dim=2)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3

class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionHyperNet(args, mode='vector')
        self.V = AttentionHyperNet(args, mode='scalar')

    def forward(self, agent_qs, inputs, imagine_groups=None):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        agent_qs = agent_qs.view(bs * max_t, 1, self.n_agents)

        w1 = self.hyper_w_1(entities, entity_mask)
        b1 = self.hyper_b_1(entities, entity_mask)
        w_final = self.hyper_w_final(entities, entity_mask)
        v = self.V(entities, entity_mask)

        w1 = w1.view(bs * max_t, self.n_agents, self.embed_dim)
        b1 = b1.view(bs * max_t, 1, self.embed_dim)
        w_final = w_final.view(bs * max_t, self.embed_dim, 1)
        v = v.view(bs * max_t, 1, 1)

        w1 = F.softmax(w1, dim=-1)
        #w_final = th.abs(w_final)
        w_final = F.softmax(w_final, dim=-1)

        hidden = th.bmm(agent_qs, w1) + b1 #(1 * na) * (na * nd)
        y = th.bmm(hidden, w_final) + v #(1 * nd) * (nd * 1)
        q_tot = y.view(bs, -1, 1)
        return q_tot

    def forward_img(self, agent_qs, inputs, imagine_groups):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        agent_qs = agent_qs.view(bs * max_t, 1, self.n_agents * 2)
        
        Wmask, Imask = imagine_groups
        w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          ne, ne))
        w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          ne, ne))
        
        w1 = th.cat([w1_W, w1_I], dim=1)
        b1 = self.hyper_b_1(entities, entity_mask)
        w_final = self.hyper_w_final(entities, entity_mask)
        v = self.V(entities, entity_mask)

        w1 = w1.view(bs * max_t, self.n_agents * 2, self.embed_dim)
        b1 = b1.view(bs * max_t, 1, self.embed_dim)
        w_final = w_final.view(bs * max_t, self.embed_dim, 1)
        v = v.view(bs * max_t, 1, 1)

        #w1 = th.abs(w1)
        w1 = F.softmax(w1, dim=-1)
        #w_final = th.abs(w_final)
        w_final = F.softmax(w_final, dim=-1)

        hidden = th.bmm(agent_qs, w1) + b1 #(1 * na) * (na * nd)
        y = th.bmm(hidden, w_final) + v #(1 * nd) * (nd * 1)
        q_tot = y.view(bs, -1, 1)
        return q_tot


