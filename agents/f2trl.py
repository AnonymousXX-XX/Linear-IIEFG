import numpy as np
from open_spiel.python import policy
import pyspiel
from numba import njit
from agents.omd import OMDBase
from agents.utils import sample_from_weights
from agents.utils import compute_log_sum_from_logit
from tqdm import tqdm
import math
from icecream import ic
import warnings 
from open_spiel.python.algorithms.exploitability import nash_conv

class F2TRL(OMDBase):
  def __init__(
      self,
      game,
      budget,
      base_constant=1.0,
      lr_constant=1.0,
      lr_pow_H=0.0,
      lr_pow_A=0.0,
      lr_pow_X=0.0,
      lr_pow_T=-0.5,
      ix_constant=1.0,
      ix_pow_H=0.0,
      ix_pow_A=0.0,
      ix_pow_X=0.0,
      ix_pow_T=-0.5,
      name=None,
      llambda=0,
      ggamma=0,
      use_cum_sta=False,
      scale_base=1,
      scale_style='',
  ):

      OMDBase.__init__(
      self,
      game,
      budget,
      base_constant=base_constant,
      lr_constant=lr_constant,
      lr_pow_H=lr_pow_H,
      lr_pow_A=lr_pow_A,
      lr_pow_X=lr_pow_X,
      lr_pow_T=lr_pow_T,
      ix_constant=ix_constant,
      ix_pow_H=ix_pow_H,
      ix_pow_A=ix_pow_A,
      ix_pow_X=ix_pow_X,
      ix_pow_T=ix_pow_T
      )
      
      self.name = 'F2TRL'
      if name:
        self.name = name
      
      self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
      self.implicit_explorations = self.base_implicit_exploration*np.ones(self.policy_shape)
      self.mu_star = np.ones(self.policy_shape)/self.A
      self.uniform_policy = np.ones(self.policy_shape)/self.A
      self.current_policy.action_probability_array = np.ones(self.policy_shape)/self.A
      self.current_logit = np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)
      self.feature_path = self.game.feature_path
      SAB_feature = np.load(self.feature_path)
      self.d = SAB_feature.shape[1]
      self.llambda = llambda
      self.ggamma = ggamma
      self.composite_feature = -SAB_feature
      self.theta_hat = np.zeros((self.H, self.d))
      self.cum_loss_hat = np.zeros_like(self.current_policy.action_probability_array).reshape((-1, 1))
      self.use_cum_statistics = use_cum_sta
      self.cum_Q_mu = [self.llambda * np.identity(self.d)] * self.H
      self.cum_experienced_XA_feature = [[]] * self.H
      self.cum_experienced_loss = [[]] * self.H
      self.cum_experienced_XAfeature_reward = [0] * self.H
      self.p_star = np.zeros(self.current_policy.action_probability_array.shape[0]) 
      self.eta_star = np.ones((self.A**(self.H+1)-1)//(self.A-1))
      self.child_dict = {}
      self.degbug_length = int(1e8)
      self.scale_base = scale_base
      for h in range(self.H):
        for x in range(self._infoset_idx(h), self._infoset_idx(h+1)):
          for a in range(self.A):
            self.child_dict[(x, a)] = [self.A*x + a + 1]
      self.compute_p_star()
      self.compute_eta_star()


  def _infoset_idx(self, h):
    return (self.A**h-1)//(self.A-1)


  def compute_policy_plan(self):
    layered_policy_list = [self.current_policy.action_probability_array[self._infoset_idx(h): self._infoset_idx(h+1), :] for h in range(self.H)]
    layered_plan_list = [np.copy(layered_policy_list[0].reshape((-1, 1)))]
    for h in range(1, self.H):
      layer_plan_tem = np.multiply(layered_plan_list[h-1], layered_policy_list[h]).reshape((-1, 1))
      layered_plan_list.append(np.copy(layer_plan_tem))
    return layered_plan_list


  def compute_loss_hat(self, layered_plan_list, composite_feature, trajectory, step = None):
    loss_hat = np.zeros_like(self.cum_loss_hat)
    XA_idx_start, XA_idx_end = None, None
    for h in range(self.H):
      if h<self.H-1:
        continue
      XA_idx_start, XA_idx_end = self.A*self._infoset_idx(h), self.A*self._infoset_idx(h+1)
      x_idx, a_idx, r = trajectory[h]['state_idx'], trajectory[h]['action_idx'], -trajectory[h]['vanilla_loss']
      phi_sqrt_mu = np.multiply(np.sqrt(layered_plan_list[h]), composite_feature[XA_idx_start: XA_idx_end])
      Q_mu = np.dot(phi_sqrt_mu.T, phi_sqrt_mu) + self.llambda * np.identity(self.d)
      theta_hat = np.dot(np.linalg.inv(Q_mu), composite_feature[x_idx * self.A + a_idx].reshape((-1, 1))) * -r
      loss_hat[XA_idx_start: XA_idx_end] = np.dot(composite_feature[XA_idx_start: XA_idx_end], theta_hat)
      self.cum_Q_mu[h] += np.dot(phi_sqrt_mu.T, phi_sqrt_mu)
      self.cum_experienced_XAfeature_reward[h] += composite_feature[x_idx * self.A + a_idx]*-r
      if self.use_cum_statistics:
        theta_hat_use_cum = np.dot(np.linalg.inv(self.cum_Q_mu[h]), self.cum_experienced_XAfeature_reward[h].reshape((-1,1)))
        loss_hat_use_cum = np.dot(composite_feature[XA_idx_start: XA_idx_end], theta_hat_use_cum)
        self.cum_loss_hat[XA_idx_start: XA_idx_end] += loss_hat_use_cum
      else:
        self.cum_loss_hat[XA_idx_start: XA_idx_end] += loss_hat[XA_idx_start: XA_idx_end]
    if step is not None and step % self.degbug_length == 0:
      ic(step, loss_hat[XA_idx_start: XA_idx_end].reshape(-1,self.A), \
        loss_hat_use_cum.reshape(-1,self.A),\
        self.cum_loss_hat.reshape(-1,self.A), self.current_policy.action_probability_array, layered_plan_list[-1])


  def compute_p_star(self):
    tem_f, tem_C = np.zeros_like(self.p_star), np.zeros_like(self.current_policy.action_probability_array)
    tem_f[self._infoset_idx(self.H-1):] = 1
    for h in range(self.H-2, -1, -1):
      for x in range(self._infoset_idx(h), self._infoset_idx(h+1)):
        for a in range(self.A):
          tem_C[x, a] = np.sum(tem_f[self.child_dict[(x, a)]])
        tem_f[x] = np.max(tem_C[x,:])
    self.p_star[0] = 1
    for h in range(self.H-1):
      for x in range(self._infoset_idx(h), self._infoset_idx(h+1)):
        for a in range(self.A):
          for child_x in self.child_dict[(x, a)]:
            self.p_star[child_x] = self.p_star[x] * tem_f[child_x] / np.sum(tem_f[self.child_dict[(x, a)]])


  def compute_eta_star(self):
    for h in range(self.H):
      self.eta_star[self._infoset_idx(h):self._infoset_idx(h+1)]=\
        self.base_learning_rate/((self.H-h)*self.p_star[self._infoset_idx(h):self._infoset_idx(h+1)])


  def update(self, step, trajectory):
    current_policy_plan = self.compute_policy_plan()
    self.compute_loss_hat(current_policy_plan, self.composite_feature, trajectory, step)
    tem_Z = np.zeros(self._infoset_idx(self.H+1))
    tem_Z[self._infoset_idx(self.H):] = 1
    tem_J = np.zeros_like(self.mu_star)
    tem_cum_loss_hat = self.cum_loss_hat.reshape((-1, self.A))
    for h in range(self.H-1, -1, -1):
      eta_ratio = self.eta_star[self._infoset_idx(h):self._infoset_idx(h+1)].reshape((-1,1))
      eta_ratio = eta_ratio/self.eta_star[self._infoset_idx(h+1):self._infoset_idx(h+2)].reshape((-1,self.A))
      tem_Z[self._infoset_idx(h+1):self._infoset_idx(h+2)] = self._truncate(tem_Z[self._infoset_idx(h+1):self._infoset_idx(h+2)],step,h)
      tem_J[self._infoset_idx(h):self._infoset_idx(h+1),:]=\
        -self.eta_star[self._infoset_idx(h):self._infoset_idx(h+1)].reshape((-1,1))*tem_cum_loss_hat[self._infoset_idx(h):self._infoset_idx(h+1),:]\
        +np.multiply(eta_ratio,np.log(tem_Z[self._infoset_idx(h+1):self._infoset_idx(h+2)].reshape((-1,self.A))))
      tem_J[self._infoset_idx(h):self._infoset_idx(h+1),:] = self._truncate(tem_J[self._infoset_idx(h):self._infoset_idx(h+1),:], step, 'tem_J_truncation')
      _tem_Z2 = np.exp(tem_J[self._infoset_idx(h):self._infoset_idx(h+1),:])
      tem_Z[self._infoset_idx(h):self._infoset_idx(h+1)]=\
        np.sum(np.multiply(self.mu_star[self._infoset_idx(h):self._infoset_idx(h+1),:], _tem_Z2),axis=1)
      _tem_Z2 /= _tem_Z2.sum(axis=1, keepdims=True)
      self.current_policy.action_probability_array[self._infoset_idx(h):self._infoset_idx(h+1),:]=\
        np.multiply(self.mu_star[self._infoset_idx(h):self._infoset_idx(h+1),:],_tem_Z2)
      _tem_normalize = np.exp(self._truncate(self.current_policy.action_probability_array[self._infoset_idx(h):self._infoset_idx(h+1),:], step, '_tem_normalize'))
      self.current_policy.action_probability_array[self._infoset_idx(h):self._infoset_idx(h+1),:] = _tem_normalize / _tem_normalize.sum(axis=1, keepdims=True)
    self.current_policy.action_probability_array = (1-self.ggamma)*self.current_policy.action_probability_array + self.ggamma*self.uniform_policy


  def _debug(self, name_list, content_list, step=None):
    debug_length=self.degbug_length
    if step is not None and step % debug_length == 0:
      for n,c in zip(name_list, content_list):
        ic(n,c)


  def _truncate(self, x, step=None, location=''):
    eps = 1e-7
    threshold = 1e7
    base = max(1.25*np.log(step+1), self.scale_base)
    y = x
    z = base*y/(np.max(np.abs(y))+eps)
    return z


  def sample_trajectory(self, step):
    plans = np.ones(self.num_players)
    trajectory = []
    state = self.game.new_initial_state()
    action = ''
    action_record = []
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = sample_from_weights(action_list, prob_list)
        action_record.append(action)
        state.apply_action(action)
      else:
        current_player = state.current_player() 
        state_idx = state.state_index()
        action, action_idx = self.sample_action_from_idx(state_idx, return_idx=True)
        action_record.append(action)
        policy = self.get_current_policy(state_idx)
        plans[current_player] *= policy[action_idx]
        transition = {
          'player': current_player,
          'state_idx': state_idx,
          'action_idx': action_idx,
          'plan': plans[current_player],
          'loss': 0.0,
          'vanilla_loss': 0,
          'true_loss': 0
        }
        trajectory += [transition]
        state.apply_action(action)
    final_returns = state.returns()
    losses = self.reward_to_loss(np.asarray(final_returns), noisy_reward=True)
    final_true_returns = state.true_returns()
    final_true_loss = self.reward_to_loss(np.asarray(final_true_returns), noisy_reward=False)
    trajectory[-1]['loss'] = losses[trajectory[-1]['player']]
    trajectory[-1]['vanilla_loss'] = -final_returns[trajectory[-1]['player']] 
    trajectory[-1]['true_loss'] = final_true_loss[trajectory[-1]['player']] 
    return trajectory