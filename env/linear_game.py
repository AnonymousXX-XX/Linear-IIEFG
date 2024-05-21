import numpy as np
import os

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
from icecream import ic


_NUM_PLAYERS = 2

_NUM_ACTIONS = int(10) # num. of actions
_NUM_STEPS = int(3) # num. of steps

# _NUM_ACTIONS = int(5) # num. of actions
# _NUM_STEPS = int(5) # num. of steps

_MAX_REWARD = 1
_NOISE_RANGE = 0.5
_DIM = 10

_GAME_TYPE = pyspiel.GameType(
    short_name="python_linear_game",
    long_name="Python Linear Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_ACTIONS,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-_MAX_REWARD,
    max_utility=_MAX_REWARD,
    utility_sum=0.0,
    max_game_length=_NUM_STEPS)


class LinearGame(pyspiel.Game):
  def __init__(self, n_actions=_NUM_ACTIONS, n_steps=_NUM_STEPS, dim=_DIM, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
    self.n_actions = n_actions
    self.n_steps = n_steps
    self.dim = dim    
    self.feature_path = './env/SAB_feature_{}_{}_{}.npy'.format(self.n_actions, self.n_steps, self.dim)
    self._generate_features_returns()
    self._max_reward = self.returns_list[0]
    self._noise_range = _NOISE_RANGE

  def new_initial_state(self):
    return LinearGameState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return LinearGameObserver(params)
  
  def _get_infoset_idx(self, h):
    return (self.n_actions**h-1)//(self.n_actions-1)
  
  def _generate_features_returns(self):
    if not os.path.exists(self.feature_path):
      SAB_feature = np.random.uniform(-1, 1, (self.n_actions*((self.n_actions**self.n_steps-1)//(self.n_actions-1)), self.dim))
      SAB_feature /= np.linalg.norm(SAB_feature, ord=2, axis=1, keepdims=True)
      np.save(self.feature_path, SAB_feature)
    else:
      SAB_feature = np.load(self.feature_path)
    theta = SAB_feature[self.n_actions*self._get_infoset_idx(self.n_steps-1),:]
    self.returns_list = np.dot(SAB_feature[self.n_actions*self._get_infoset_idx(self.n_steps-1):,:], theta.reshape((-1, 1))).squeeze()
    ic(self.returns_list.reshape((-1,self.n_actions)))

class LinearGameState(pyspiel.State):
  def __init__(self, game):
    super().__init__(game)
    self._is_terminal = False
    self._current_step = 0
    self._game_over = False
    self._next_player = 0
    self._p0_infoset_list = []
    self.n_actions = game.n_actions
    self.n_steps = game.n_steps
    self.state_str = ''
    self.returns_list = game.returns_list

  def current_player(self):
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._next_player

  def _legal_actions(self, player):
    return list(np.arange(self.n_actions))

  def _apply_action(self, action):
    self._p0_infoset_list.append(str(action))
    self.state_str = '_'.join(self._p0_infoset_list)
    self._current_step += 1
    self._is_terminal = (self._current_step == self.n_steps)

  def _action_to_string(self, player, action):
    return str(action)

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    p0_final_state = 0
    for i in range(self.n_steps):
      p0_final_state += int(self._p0_infoset_list[self.n_steps-1-i]) * (self.n_actions**i)
    return self.returns_list[p0_final_state]+np.random.uniform(low=-_NOISE_RANGE, high=_NOISE_RANGE, size=1), -self.returns_list[p0_final_state]+np.random.uniform(low=-_NOISE_RANGE, high=_NOISE_RANGE, size=1)
  
  def true_returns(self):
    p0_final_state = 0
    for i in range(self.n_steps):
      p0_final_state += int(self._p0_infoset_list[self.n_steps-1-i]) * (self.n_actions**i)
    return self.returns_list[p0_final_state], -self.returns_list[p0_final_state]

  def __str__(self):
    return self.state_str
  
  def information_state_string(self, player = None):
    return self.state_str
  
  def _get_infoset_idx(self, h):
    return (self.n_actions**h-1)//(self.n_actions-1)

  def state_index(self):
    idx = 0
    l = len(self._p0_infoset_list)
    for i in range(l):
      idx += int(self._p0_infoset_list[l-1-i]) * (self.n_actions**i)
    return idx+self._get_infoset_idx(l)


class LinearGameObserver:
  def __init__(self, params):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self.state_str = ''

  def set_from(self, state, player):
    self.state_str = state.state_str

  def string_from(self, state, player):
    return state.state_str


pyspiel.register_game(_GAME_TYPE, LinearGame)
