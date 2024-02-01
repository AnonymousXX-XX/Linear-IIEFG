import numpy as np
import math
from numba import njit

from open_spiel.python import policy
import pyspiel

from open_spiel.python.algorithms.exploitability import nash_conv
from agents.utils import sample_from_weights
from agents.utils import compute_log_sum_from_logit
from tqdm import tqdm

from icecream import ic

@njit
def _update_current_policy(
  state_idx,
  logits,
  legal_actions_mask,
  action_probability_array
):
  probs = np.exp(logits[state_idx,:])*legal_actions_mask[state_idx,:]
  probs /= probs.sum()
  action_probability_array[state_idx,:] = probs


class OMDBase(object):
  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    lr_pow_H=0.0,
    lr_pow_A=0.0,
    lr_pow_X=0.0,
    lr_pow_T=0.0,
    ix_constant=1.0,
    ix_pow_H=0.0,
    ix_pow_A=0.0,
    ix_pow_X=0.0,
    ix_pow_T=0.0
  ):
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "OMD requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")
    self.game = game
    self.num_players = game.num_players()
    assert self.num_players < 3, "OMD not implemented for more than 2 players"
    self.range_rewards = [self.game.min_utility(), self.game.max_utility()]
    self.range_noisy_rewards = [self.game.min_utility()-self.game._noise_range, self.game.max_utility()+self.game._noise_range]
    self.budget = budget

    self.eps = 1e-8
    self.current_policy = policy.TabularPolicy(game)
    self.average_policy = self.current_policy.__copy__()
    self.policy_shape = self.current_policy.action_probability_array.shape
    self.action_from_idx=np.zeros(self.policy_shape[1])
    self.legal_actions_mask = np.array(
      self.current_policy.legal_actions_mask.copy(),
       dtype=bool)
    self.cumulative_plan = np.zeros(self.policy_shape)

    self.H = self.game.max_game_length()
    self.A = self.game.num_distinct_actions()
    self.compute_base_rates_and_policies(
      budget,
      base_constant,
      lr_constant,
      lr_pow_H,
      lr_pow_A,
      lr_pow_X,
      lr_pow_T,
      ix_constant,
      ix_pow_H,
      ix_pow_A,
      ix_pow_X,
      ix_pow_T
      )
    
    self.cum_loss = 0
    self.cum_vanilla_loss = 0
    
  def compute_base_rates_and_policies(
    self,
    budget,
    base_constant,
    lr_constant,
    lr_pow_H,
    lr_pow_A,
    lr_pow_X,
    lr_pow_T,
    ix_constant,
    ix_pow_H,
    ix_pow_A,
    ix_pow_X,
    ix_pow_T
    ):
    X = self.policy_shape[0]/2
    A = self.policy_shape[1]
    H = self.game.max_game_length()
    self.base_learning_rate = H**lr_pow_H * A**lr_pow_A * X**lr_pow_X * budget**lr_pow_T * lr_constant * base_constant
    self.base_implicit_exploration = H**ix_pow_H * A**ix_pow_A * X**ix_pow_X * budget**ix_pow_T * ix_constant * base_constant
    self.legal_actions_indicator=self.legal_actions_mask
    self.number_actions_from_idx=self.legal_actions_indicator.sum(axis=-1)
    self.uniform_policy=self.legal_actions_indicator/self.number_actions_from_idx[:,np.newaxis]
  
  def reward_to_loss(self,reward,noisy_reward=False):
    if not noisy_reward:
      loss = (self.range_rewards[1]-reward)
      loss /= (self.range_rewards[1]-self.range_rewards[0])
    else:
      loss = (self.range_noisy_rewards[1]-reward)
      loss /= (self.range_noisy_rewards[1]-self.range_noisy_rewards[0])
    return loss

  def state_index(self, state):
    return self.current_policy.state_index(state)
    
  def sample_action(self, state, return_idx=False):
    policy = self.current_policy(state)
    probs = list(policy.values())
    action_idx = sample_from_weights(list(range(len(policy))),probs)
    action = list(policy.keys())[action_idx]
    if return_idx:
      return action, action_idx
    return action

  def sample_action_from_idx(self, state_idx, return_idx=False):
    probs = self.current_policy.action_probability_array[state_idx,:]
    action_idx = sample_from_weights(list(range(probs.shape[0])), probs)
    action=action_idx
    if return_idx:
      return action, action_idx
    return action
  
  def get_current_policy(self, state_idx):
      return self.current_policy.action_probability_array[state_idx,:]
  
  def set_current_policy(self, state_idx, new_policy):
      self.current_policy.action_probability_array[state_idx,:]=new_policy
      
  def update_current_policy(self, state_idx):
    if state_idx is None:
      self.current_policy.action_probability_array = np.exp(self.logits)*self.legal_actions_mask
      self.current_policy.action_probability_array /= self.current_policy.action_probability_array.sum(axis=-1, keepdims=True)
    else:
      _update_current_policy(
        state_idx,
        self.logits,
        self.legal_actions_mask,
        self.current_policy.action_probability_array
      )

  def update_average_policy(self):
    self.average_policy.action_probability_array = self.cumulative_plan * self.legal_actions_mask +self.eps
    self.average_policy.action_probability_array /= self.average_policy.action_probability_array.sum(axis=-1, keepdims=True)

  def sample_trajectory(self, step):
    plans = np.ones(self.num_players)
    cum_plans = np.ones(self.num_players)*(step+1.0)
    trajectory = []
    state = self.game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = sample_from_weights(action_list, prob_list)
        state.apply_action(action)
      else:
        current_player = state.current_player() 
        state_idx = self.state_index(state)
        action, action_idx = self.sample_action_from_idx(state_idx, return_idx=True)
        policy = self.get_current_policy(state_idx)
        self.cumulative_plan[state_idx,:] += (cum_plans[current_player]-self.cumulative_plan[state_idx,:].sum())* policy
        cum_plans[current_player] = self.cumulative_plan[state_idx, action_idx]
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

  def update(self, step, trajectory):
    values =  np.zeros(self.num_players)
    for transition in reversed(trajectory):
      player, state_idx, action_idx, plan, loss, vanilla_loss, true_loss = transition.values()
      policy = self.current_policy.action_probability_array[state_idx,:]
      ix = self.implicit_explorations[state_idx, action_idx]
      ix_loss = loss/(plan+ix)
      legal_actions=self.legal_actions_indicator[state_idx,:]
      lr=self.learning_rates[state_idx]
      adjusted_loss = ix_loss - values[player]
      self.current_logit[state_idx,action_idx]-=lr*adjusted_loss
      logz=compute_log_sum_from_logit(self.current_logit[state_idx,:],legal_actions)
      self.current_logit[state_idx,:]-=logz*legal_actions
      values[player] = logz/lr
      new_policy=np.exp(self.current_logit[state_idx,:],where=legal_actions)*legal_actions
      self.set_current_policy(state_idx, new_policy)
      
  def fit(
    self,
    new_scale=None,
    fix_log_interval=None,
    log_interval=None,
    writer_path=None,
    record_exploitabilities = False,
    record_current = False,
    verbose=1,
    number_points=None,
    first_point=None,
    wandb_obj=None,
    wandb_interval_log=False,
  ):
    self.writer_path = writer_path
    if writer_path is not None:
        self.writer = None
    else:
        self.writer = None
    if record_exploitabilities:
      list_exploitability = {'step':[], 'current':[], 'average':[]}
    if fix_log_interval and log_interval is not None:
      recorded_steps=[step for step in range(self.budget-1,0,-log_interval)]
    if not fix_log_interval and number_points is not None:
      log_step=(math.log(self.budget)-math.log(first_point))/(number_points-1)
      recorded_steps=[self.budget-1]+[round(first_point*math.exp(i*log_step))-1 for i in range(number_points-2,0,-1)]+[first_point-1]
    cur_loss = 0
    for step in tqdm(range(self.budget), disable=(verbose < 1 )):
      trajectory = self.sample_trajectory(step)
      cur_loss = trajectory[-1]['true_loss']
      self.cum_loss += cur_loss
      self.cum_vanilla_loss += trajectory[-1]['vanilla_loss']
      self.update(step, trajectory)
      if wandb_obj is not None:
        if not wandb_interval_log:
          wandb_obj.log({"cum_regret": self.cum_loss})
          if record_current:
            wandb_obj.log({"episode_regret": cur_loss})
        elif fix_log_interval and log_interval is not None:
          if type(self.game).__name__ == 'LinearGame':
            if step%log_interval == 0:
                log_dict = {
                    "cum_regret": self.cum_loss,
                    "custom_step": step+1   
                }
                if record_current:
                  log_dict["episode_regret"] = cur_loss
                wandb_obj.log(log_dict)
        else:
          if step%log_interval == 0:
              self.update_average_policy()
              exploit_average = nash_conv(self.game, self.average_policy, use_cpp_br=True)
              log_dict = {
                  "explo_avg": exploit_average,
                  "custom_step": step+1   
              }
              if record_current:
                exploit_current = nash_conv(self.game, self.current_policy, use_cpp_br=True)
                log_dict["explo_cur"] = exploit_current
              wandb_obj.log(log_dict)
      else:
        if step==recorded_steps[-1]:
          if type(self.game).__name__ == 'LinearGame':
            if record_current:
              exploit_current = cur_loss
            exploit_average = self.cum_loss
          else:
            if record_current:
              exploit_current = nash_conv(self.game, self.current_policy, use_cpp_br=True)
            self.update_average_policy()
            exploit_average = nash_conv(self.game, self.average_policy, use_cpp_br=True)
          if record_exploitabilities:
            list_exploitability['step'].append(step+1)
            list_exploitability['average'].append(exploit_average)
            if record_current:
              list_exploitability['current'].append(exploit_current)
          if self.writer:
            if record_current:
              self.writer.add_scalar('exploitability/current', exploit_current, step)
            self.writer.add_scalar('exploitability/average', exploit_average, step)
          if verbose > 1:
            tqdm.write('')
            tqdm.write(f'step: {step}')
            if record_current:
              tqdm.write(f'exploitability current: {exploit_current}')
            tqdm.write(f'exploitability average: {exploit_average}')
          recorded_steps.pop()
    if record_exploitabilities:
      list_exploitability['step'] = np.array(list_exploitability['step'])
      if record_current:
        list_exploitability['current'] = np.array(list_exploitability['current'])
      list_exploitability['average'] = np.array(list_exploitability['average'])
      return list_exploitability


    









   

