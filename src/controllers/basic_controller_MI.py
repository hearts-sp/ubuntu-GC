from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC_MI:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states_q = None
        self.hidden_states_pi_avg = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ret_agent_outs=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        _, pi, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(pi[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, q_vals[bs], pi[bs], pi_avg[bs]
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        outputs, hidden_states = self.agent(agent_inputs, 
                                            self.hidden_states_q,
                                            self.hidden_states_pi_avg)
        
        q_vals, logit, logit_avg = outputs
        self.hidden_states_q, self.hidden_states_pi_avg = hidden_states

        pi = th.nn.functional.softmax(logit, dim=-1)
        pi_avg = th.nn.functional.softmax(logit_avg, dim=-1)

        if not test_mode:
            epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()
            random_pi = th.ones_like(pi)
            random_pi = random_pi / epsilon_action_num
            eps = self.action_selector.epsilon
            pi = (1 - eps) * pi + random_pi * eps
            pi_avg = (1 - eps) * pi_avg + random_pi * eps
            
        if int_t:
            return q_vals.squeeze(1), pi.squeeze(1), pi_avg.squeeze(1)

        return q_vals, pi, pi_avg

    def forward_img(self, ep_batch, t, test_mode=False, **kwargs):
        t = slice(0, ep_batch["avail_actions"].shape[1])

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        avail_actions = avail_actions.repeat(2, 1, 1, 1)

        outputs, groups = self.agent.forward_img(agent_inputs,
                                                 self.hidden_states_q_img,
                                                 self.hidden_states_pi_avg_img)

        q_vals, logit, logit_avg = outputs

        pi = th.nn.functional.softmax(logit, dim=-1)
        pi_avg = th.nn.functional.softmax(logit_avg, dim=-1)

        if not test_mode:
            epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()
            random_pi = th.ones_like(pi)
            random_pi = random_pi / epsilon_action_num
            eps = self.action_selector.epsilon
            pi = (1 - eps) * pi + random_pi * eps
            pi_avg = (1 - eps) * pi_avg + random_pi * eps

        return (q_vals, pi, pi_avg), groups

    def init_hidden(self, batch_size):
        hidden_states_q, hidden_states_pi_avg, hidden_states_q_img, hidden_states_pi_avg_img = self.agent.init_hidden()
        self.hidden_states_q = hidden_states_q.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states_pi_avg = hidden_states_pi_avg.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states_q_img = hidden_states_q_img.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states_pi_avg_img = hidden_states_pi_avg_img.unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.__parameters__()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def eval(self):
        self.agent.eval()

    def train(self):
        self.agent.train()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs, ts, na, os = batch["obs"].shape
        inputs = []
        inputs.append(batch["obs"][:, t])  # btav
        if self.args.obs_last_action:
            if t.start == 0:
                acs = th.zeros_like(batch["actions_onehot"][:, t])
                acs[:, 1:] = batch["actions_onehot"][:, slice(0, t.stop - 1)]
            else:
                acs = batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)]
            inputs.append(acs)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t.stop - t.start, -1, -1))
        inputs = th.cat(inputs, dim=3)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
