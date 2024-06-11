import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer_MI

class EntityAttentionRNNModule(nn.Module):
    def __init__(self, input_shape, args, bool_avg=False):
        super(EntityAttentionRNNModule, self).__init__()
        self.args = args
        self.bool_avg = bool_avg
        input_shape_entity = input_shape
        self.fc1 = nn.Linear(input_shape_entity, args.attn_embed_dim)
        if not self.bool_avg:
            self.attn = EntityAttentionLayer_MI(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)

        self.fc2 = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)


        self.fc_q = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.fc_pi = nn.Linear(args.rnn_hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents] # bs * ts, na

        x1 = F.relu(self.fc1(entities))
        if not self.bool_avg:
            attn_outs = self.attn(x1, 
                pre_mask=obs_mask,
                post_mask=agent_mask, 
                ret_attn_logits=False)
            x2 = attn_outs
        else:
            x2 = x1[:, :self.args.n_agents]
        x3 = F.relu(self.fc2(x2))
        x3 = x3.reshape(bs, ts, self.args.n_agents, -1)
        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x3 = x3[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x3, h)
            hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time
        output_q = self.fc_q(hs)
        output_pi = self.fc_pi(hs)

        output_q = output_q.reshape(bs, ts, self.args.n_agents, -1)
        output_q = output_q.masked_fill(agent_mask.bool().reshape(bs, ts, self.args.n_agents, 1), 0)

        output_pi = output_pi.reshape(bs, ts, self.args.n_agents, -1)
        output_pi = output_pi.masked_fill(agent_mask.bool().reshape(bs, ts, self.args.n_agents, 1), -1e10)

        return (output_q, output_pi), hs



class EntityAttentionRNNAgent_MI(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionRNNAgent_MI, self).__init__()
        self.args = args
        input_shape_entity = input_shape
        self.q_net = EntityAttentionRNNModule(input_shape_entity, args, bool_avg=False)
        self.pi_avg_net = EntityAttentionRNNModule(input_shape_entity, args, bool_avg=True)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.q_net.init_hidden(), self.pi_avg_net.init_hidden(), self.q_net.init_hidden(), self.pi_avg_net.init_hidden()

    def __parameters__(self):
        return self.q_net.parameters(), self.pi_avg_net.parameters()

    def forward(self, inputs, hidden_state_q, hidden_state_pi_avg):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape

        agent_mask = entity_mask[:, :, :self.args.n_agents]

        input_q = (entities, obs_mask, entity_mask)
        input_pi_avg = (entities, obs_mask, entity_mask)


        (q, pi), h_q = self.q_net(input_q, hidden_state_q)
        (_, pi_avg), h_pi_avg = self.pi_avg_net(input_pi_avg, hidden_state_pi_avg)

        return (q, pi, pi_avg), (h_q, h_pi_avg)

    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def entitymask2attnmask(self, entity_mask):
        bs, ts, ne = entity_mask.shape
        # agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - entity_mask.to(th.float)).reshape(bs * ts, ne, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)

    def forward_img(self, inputs, hidden_state_q, hidden_state_pi_avg, **kwargs):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)
        groupB = self.logical_not(groupA)
        # mask out entities not present in env
        groupA = self.logical_or(groupA, entity_mask[:, [0]])
        groupB = self.logical_or(groupB, entity_mask[:, [0]])

        # convert entity mask to attention mask
        groupAattnmask = self.entitymask2attnmask(groupA)
        groupBattnmask = self.entitymask2attnmask(groupB)
        # create attention mask for interactions between groups
        interactattnmask = self.logical_or(self.logical_not(groupAattnmask),
                                           self.logical_not(groupBattnmask))
        # get within group attention mask
        withinattnmask = self.logical_not(interactattnmask)

        activeattnmask = self.entitymask2attnmask(entity_mask[:, [0]])
        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self.logical_or(withinattnmask, activeattnmask)
        Iattnmask_noobs = self.logical_or(interactattnmask, activeattnmask)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self.logical_or(withinattnmask, obs_mask)
        interactattnmask = self.logical_or(interactattnmask, obs_mask)

        entities = entities.repeat(2, 1, 1, 1)
        obs_mask = th.cat([withinattnmask, interactattnmask], dim=0)
        entity_mask = entity_mask.repeat(2, 1, 1)

        inputs_q = (entities, obs_mask, entity_mask)
        hidden_state_q = hidden_state_q.repeat(2, 1, 1)
        (q, pi), _ = self.q_net(inputs_q, hidden_state_q)

        input_pi_avg = (entities, obs_mask, entity_mask)
        hidden_state_pi_avg = hidden_state_pi_avg.repeat(2, 1, 1)
        (_, pi_avg), _ = self.pi_avg_net(input_pi_avg, hidden_state_pi_avg)

        return (q, pi, pi_avg), (Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1))
