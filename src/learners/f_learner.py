import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.flex_qmix_MI import FlexQMixer
import torch as th
from torch.optim import RMSprop
from torch.distributions import Categorical
import torch.nn.functional as F

class FLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        #q_param, pi_param, pi_avg_param = mac.parameters()
        q_param, pi_avg_param = mac.parameters()
        self.q_param = list(q_param)
        self.pi_avg_param = list(pi_avg_param)

        self.last_target_update_episode = 0
        self.ba_iters = 4

        self.alpha_start = args.alpha_start
        self.alpha_end = args.alpha_end
        self.alpha_decay_end = args.alpha_anneal_time

        self.alpha_step = (self.alpha_end - self.alpha_start) / self.alpha_decay_end
        self.lmbda = args.lmbda

        self.debug = args.learner_debug
        self.debug_type = getattr(args, "debug_type", "MI")

        if args.mixer == "flex_qmix_MI":
            self.mixer = FlexQMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        self.q_param += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.q_optimiser = RMSprop(params=self.q_param, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)
        
        self.pi_avg_optimiser = RMSprop(params=self.pi_avg_param, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def _get_mixer_ins(self, batch, repeat_batch=1):
        entities = []
        bs, max_t, ne, ed = batch["entities"].shape
        entities.append(batch["entities"])
        if self.args.entity_last_action:
            last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                    device=batch.device,
                                    dtype=batch["entities"].dtype)
            last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
            entities.append(last_actions)

        entities = th.cat(entities, dim=3)
        return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                (entities[:, 1:],
                batch["entity_mask"][:, 1:]))

    def train_critic(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        #avail_agents= batch["entity_mask"]

        if self.alpha_step < 0 :
            alpha = max(self.alpha_end, self.alpha_start + t_env * self.alpha_step) # linear decay
        else:
            alpha = min(self.alpha_end, self.alpha_start + t_env * self.alpha_step) # linear increase

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()

        q_vals, pi_out, pi_avg_out = self.mac.forward(batch, t=None)
        chosen_action_qvals = th.gather(q_vals[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        (img_qvals, _, _), groups = self.mac.forward_img(batch, t=None)
        rep_actions = actions.repeat(2, 1, 1, 1)
        img_chosen_action_qvals = th.gather(img_qvals[:, :-1], dim=3, index=rep_actions).squeeze(3)

        W_qvals, I_qvals = img_qvals.chunk(2, dim=0)
        caqW, caqI = img_chosen_action_qvals.chunk(2, dim=0)
        caq_imagine = th.cat([caqW, caqI], dim=2)

        self.target_mac.init_hidden(batch.batch_size)
        target_qvals, _, _ = self.target_mac.forward(batch, t=None)

        target_qvals = target_qvals.detach()[:, 1:]
        pi = pi_out.detach()
        pi_avg = pi_avg_out.detach()

        pi[avail_actions == 0] = 1e-10
        pi = pi / pi.sum(dim=-1, keepdim=True)
        pi[avail_actions == 0] = 1e-10

        pi_avg[avail_actions == 0] = 1e-10
        pi_avg = pi_avg / pi_avg.sum(dim=-1, keepdim=True)
        pi_avg[avail_actions == 0] = 1e-10

        
        next_actions = Categorical(probs=pi).sample().long().unsqueeze(3)
        target_chosen_qvals = th.gather(target_qvals, dim=3, index=next_actions[:, 1:]).squeeze(3)
        
        pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:, 1:]
        pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
        log_pi_taken = pi_taken.log()
        
        pi_avg_taken = th.gather(pi_avg, dim=3, index=next_actions).squeeze(3)[:, 1:]
        pi_avg_taken[mask.expand_as(pi_taken) == 0] = 1.0
        log_pi_avg_taken = pi_avg_taken.log()



        mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
        chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
        groups = [gr[:, :-1] for gr in groups]
        caq_imagine = self.mixer.forward_img(caq_imagine, mix_ins, groups)
        target_chosen_qvals = self.target_mixer(target_chosen_qvals, targ_mix_ins)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_chosen_qvals
        targets = targets - alpha * (log_pi_taken - log_pi_avg_taken).mean(dim=-1, keepdim=True)

        KL_critic = alpha * (log_pi_taken - log_pi_avg_taken).mean(dim=-1, keepdim=True)
        max_KL_critic = th.max(alpha * (log_pi_taken - log_pi_avg_taken))
        H_critic = alpha * log_pi_taken.mean(dim=-1, keepdim=True)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        critic_loss = (masked_td_error ** 2).sum() / mask.sum()

        im_td_error = (caq_imagine - targets.detach())
        im_masked_td_error = im_td_error * mask
        im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
        # Normal L2 loss, take mean over actual data
        loss = critic_loss * self.lmbda +  im_loss * (1 - self.lmbda)
        # Optimise
        self.q_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.q_param, self.args.grad_norm_clip)
        self.q_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic loss", critic_loss.item(), t_env)
            try:
                self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            except:
                self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("alpha_critic", alpha, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("KL_critic", (KL_critic * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("H_critic", (H_critic * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("max_KL_critic", max_KL_critic.item(), t_env)

    def train_actor_avg(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        #torch.autograd.set_detect_anomaly(True)
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).unsqueeze(-1)
        img_mask = mask.repeat(2, 1, 1, 1)
        avail_actions = batch["avail_actions"]
        img_avail_actions = batch["avail_actions"].repeat(2, 1, 1, 1)

        if self.alpha_step < 0 :
            alpha = max(self.alpha_end, self.alpha_start + t_env * self.alpha_step) # linear decay
        else:
            alpha = min(self.alpha_end, self.alpha_start + t_env * self.alpha_step) # linear increase

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()

        _, pi_out, pi_avg_out = self.mac.forward(batch, t=None, ret_rep=False)
        
        pi_out = pi_out.detach()
        pi_target = pi_out.detach()
        
        pi_target[avail_actions == 0] = 1e-10
        pi_target = pi_target / pi_target.sum(dim=-1, keepdim=True)
        pi_target[avail_actions == 0] = 1.0

        pi_target = pi_target.clone()[:, :-1]
        pi_target[mask.expand_as(pi_target) == 0] = 1.0
        log_pi_target = pi_target.log()

        pi_avg_out[avail_actions == 0] = 1e-10
        pi_avg_out = pi_avg_out / pi_avg_out.sum(dim=-1, keepdim=True)
        pi_avg_out[avail_actions == 0] = 1.0

        pi_avg = pi_avg_out.clone()[:, :-1]
        pi_avg[mask.expand_as(pi_avg) == 0] = 1.0
        log_pi_avg = pi_avg.log()

        (_, img_pi_out, img_pi_avg_out), _ = self.mac.forward_img(batch, t=None)

        img_pi_out = img_pi_out.detach()
        img_pi_target = img_pi_out.detach()

        img_pi_target[img_avail_actions == 0] = 1e-10
        img_pi_target = img_pi_target / img_pi_target.sum(dim=-1, keepdim=True)
        img_pi_target[img_avail_actions == 0] = 1.0

        img_pi_target = img_pi_target.clone()[:, :-1]
        img_pi_target[img_mask.expand_as(img_pi_target) == 0] = 1.0
        img_log_pi = img_pi_target.log()

        img_pi_avg_out[img_avail_actions == 0] = 1e-10
        img_pi_avg_out = img_pi_avg_out / img_pi_avg_out.sum(dim=-1, keepdim=True)
        img_pi_avg_out[img_avail_actions == 0] = 1.0

        img_pi_avg = img_pi_avg_out.clone()[:, :-1]
        img_pi_avg[img_mask.expand_as(img_pi_avg) == 0] = 1.0
        img_log_pi_avg = img_pi_avg.log()


        #pi_avg_target = (pi_avg * (log_pi_avg - log_pi.detach()))
        pi_avg_target = -(pi_target * log_pi_avg)
        img_pi_avg_target = -(img_pi_target * img_log_pi_avg)

        max_kl_avg = th.max(log_pi_avg - log_pi_target)

        pi_avg_loss = (pi_avg_target * mask).sum() / mask.sum()
        img_pi_avg_loss = (img_pi_avg_target * img_mask).sum() / img_mask.sum()
        
        loss = pi_avg_loss * (self.lmbda / self.ba_iters)  + img_pi_avg_loss * (1 - self.lmbda)


        # Optimise
        self.pi_avg_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.pi_avg_param, self.args.grad_norm_clip)
        self.pi_avg_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("pi_avg_loss", pi_avg_loss.item(), t_env)
            self.logger.log_stat("max_KL_avg", max_kl_avg.item(), t_env)
            #self.logger.log_stat("rep_loss", rep_loss.item(), t_env)
            try:
                self.logger.log_stat("pi_avg_grad_norm", grad_norm.item(), t_env)
            except:
                self.logger.log_stat("pi_avg_grad_norm", grad_norm, t_env)

    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).unsqueeze(-1)
        img_mask = mask.repeat(2, 1, 1, 1)
        avail_actions = batch["avail_actions"]
        img_avail_actions = batch["avail_actions"].repeat(2, 1, 1, 1)

        if self.alpha_step < 0 :
            alpha = max(self.alpha_end, self.alpha_start + t_env * self.alpha_step) # linear decay
        else:
            alpha = min(self.alpha_end, self.alpha_start + t_env * self.alpha_step) # linear increase

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()

        q_vals, pi_out, pi_avg_out = self.mac.forward(batch, t=None)

        q_vals = q_vals.detach()
        pi_avg_out = pi_avg_out.detach()
        q_vals = q_vals[:, :-1]

        pi_out[avail_actions == 0] = 1e-10
        pi_out = pi_out / pi_out.sum(dim=-1, keepdim=True)
        pi_out[avail_actions == 0] = 1.0
        pi = pi_out.clone()[:, :-1]
        pi[mask.expand_as(pi) == 0] = 1.0
        log_pi = pi.log()

        pi_avg_out[avail_actions == 0] = 1e-10
        pi_avg_out = pi_avg_out / pi_avg_out.sum(dim=-1, keepdim=True)
        pi_avg_out[avail_actions == 0] = 1.0
        pi_avg = pi_avg_out.clone()[:, :-1]
        pi_avg[mask.expand_as(pi_avg) == 0] = 1.0
        log_pi_avg = pi_avg.log()

        (img_q_vals, img_pi_out, img_pi_avg_out), _ = self.mac.forward_img(batch, t=None)

        img_q_vals = img_q_vals.detach()
        img_pi_avg_out = img_pi_avg_out.detach()
        img_q_vals = img_q_vals[:, :-1]


        img_pi_out[img_avail_actions == 0] = 1e-10
        img_pi_out = img_pi_out / img_pi_out.sum(dim=-1, keepdim=True)
        img_pi_out[img_avail_actions == 0] = 1.0
        img_pi = img_pi_out.clone()[:, :-1]
        img_pi[img_mask.expand_as(img_pi) == 0] = 1.0
        img_log_pi = img_pi.log()

        img_pi_avg_out[img_avail_actions == 0] = 1e-10
        img_pi_avg_out = img_pi_avg_out / img_pi_avg_out.sum(dim=-1, keepdim=True)
        img_pi_avg_out[img_avail_actions == 0] = 1.0
        img_pi_avg = img_pi_avg_out.clone()[:, :-1]
        img_pi_avg[img_mask.expand_as(img_pi_avg) == 0] = 1.0
        img_log_pi_avg = img_pi_avg.log()

        KL_actor = (pi * (log_pi - log_pi_avg)).sum(dim=-1)
        H_actor = (pi * (log_pi)).sum(dim=-1)

        max_kl_actor = th.max(log_pi - log_pi_avg)

        pi_target = pi * (alpha * (log_pi - log_pi_avg.detach()) - q_vals)
        img_pi_target = img_pi * (alpha * (img_log_pi - img_log_pi_avg.detach()) - img_q_vals)

        pi_loss = (pi_target * mask).sum() / mask.sum()
        img_pi_loss = (img_pi_target * img_mask).sum() / img_mask.sum()
        loss = pi_loss * self.lmbda + img_pi_loss * (1 - self.lmbda)
        

        # Optimise
        self.q_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.q_param, self.args.grad_norm_clip)
        self.q_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("alpha_actor", alpha, t_env)
            self.logger.log_stat("pi_loss", pi_loss.item(), t_env)
            self.logger.log_stat("KL_actor", KL_actor.mean().item(), t_env)
            self.logger.log_stat("H_actor", H_actor.mean().item(), t_env)
            self.logger.log_stat("max_KL_actor", max_kl_actor.item(), t_env)
            try:
                self.logger.log_stat("pi_grad_norm", grad_norm.item(), t_env)
            except:
                self.logger.log_stat("pi_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.train_critic(batch, t_env, episode_num)
        for _ in range(self.ba_iters):
            self.train_actor_avg(batch, t_env, episode_num)
            self.train_actor(batch, t_env, episode_num)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)

    def load_models(self, path, evaluate=False):
        self.mac.load_models(path)

