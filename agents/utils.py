import os
import importlib
from datetime import datetime
from collections import defaultdict
from unicodedata import name
import yaml
import pickle
import random
import math
import numpy as np
import pyspiel
import ray

import wandb
from icecream import ic

from env.linear_game import LinearGame

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def sample_from_weights(population, weights):
  return random.choices(population, weights=weights)[0] if np.sum(weights)>0 else random.choices(population)[0]

def compute_log_sum_from_logit(logit,mask):
  logit_max=logit.max(initial=-np.inf,where=mask)
  return math.log(np.sum(np.exp(logit-logit_max,where=mask),where=mask))+logit_max

def get_class(class_name):
  module_names = class_name.split('.')
  module = ".".join(module_names[:-1])          
  return getattr(importlib.import_module(module), module_names[-1])



class ExperimentGenerator(object):
  def __init__(
    self,
    args,
    description,
    game_names,
    agents,
    save_path,
    global_init_kwargs=None,
    global_training_kwargs=None,
    tuning_parameters=None,
    n_simulations=4,
    wandb_args=None,
    seed_list=None,
  ):
    self.args = args
    self.description = description
    self.game_names = game_names
    self.n_simulations = n_simulations
    self.wandb_args = wandb_args
    self.seed_list = seed_list
    self.global_init_kwargs = {}
    if global_init_kwargs:
        self.global_init_kwargs = global_init_kwargs
    self.training_kwargs = {}
    if global_training_kwargs:
        self.training_kwargs = global_training_kwargs
    self.tuning_parameters = tuning_parameters
    self.tuned_rates = None
    self.save_path = os.path.join(save_path, description)
    self.dict_agent_constructor = {}
    self.dict_agent_kwargs = {}
    self.agent_names = []
    for agent_config_path in agents:
      agent_config = yaml.load(open(agent_config_path, 'r'), Loader=yaml.FullLoader)
      agent_class_name = agent_config['agent_class']
      agent_class = get_class(agent_class_name)
      agent_kwargs = agent_config['init_kwargs']
      if self.global_init_kwargs:
        for key, value in self.global_init_kwargs.items():
          agent_kwargs[key] = value
      agent_name = agent_kwargs['name']
      self.agent_names.append(agent_name)
      self.dict_agent_kwargs[agent_name] = agent_kwargs
      self.dict_agent_constructor[agent_name] = agent_class


  def save_results(self, results, game_name, agent_name):
    now = datetime.now().strftime("%d-%m__%H:%M")
    save_path = os.path.join(self.save_path, game_name, agent_name, now+'.pickle')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as _f:
      pickle.dump(results, _f)

  
  def load_results(self):
    dict_results = {}
    for game_name in self.game_names:
        dict_results[game_name]={}
        for agent_name in self.agent_names:
            save_path = os.path.join(self.save_path, game_name, agent_name)
            list_res = os.listdir(save_path)
            latest_res = max(list_res)
            save_path = os.path.join(save_path, latest_res)
            with open(save_path, 'rb') as _f:
                dict_results[game_name][agent_name] = pickle.load(_f)
    return dict_results


  def run_wo_ray(self):
    game_name_str = get_game_name_str(self.args.game)
    if self.tuned_rates is None:
        base_constant=1.0
    else:
        base_constant=self.tuned_rates[game_name_str][self.args.algo]
    fit_agent_wo_remote(
      self.dict_agent_constructor[self.args.algo],
      self.dict_agent_kwargs[self.args.algo],
      self.args.game,
      base_constant,
      self.training_kwargs,
      self.wandb_args,
      self.seed_list[self.args.seed_idx]
    )
    print('Finished!')


  def run(self):
    list_tasks = []
    for game_name in self.game_names:
        game_name_str = get_game_name_str(game_name)
        for agent_name in self.agent_names:
            for _ in range(self.n_simulations):
                seed = self.seed_list[_]
                setup_seed(seed)
                if self.tuned_rates is None:
                    base_constant=1.0
                else:
                    base_constant=self.tuned_rates[game_name_str][agent_name]
                list_tasks.append([
                    self.dict_agent_constructor[agent_name],
                    self.dict_agent_kwargs[agent_name],
                    game_name,
                    base_constant,
                    self.training_kwargs,
                    self.wandb_args,
                    seed
                    ])
    ray.init()
    result_ids = []
    for task in list_tasks:
       result_ids.append(fit_agent.remote(*task))
    results = ray.get(result_ids)
    ray.shutdown()
    print('Finished!')
    idx = 0
    for game_name in self.game_names:
        game_name_str = get_game_name_str(game_name)
        for agent_name in self.agent_names:
            final_results = defaultdict(list)
            for _ in range(self.n_simulations):
                res = results[idx]
                for key, value in res.items():
                    final_results[key].append(value)
                idx+=1
            for key in final_results.keys():
                if key == 'step':
                    final_results[key] = final_results[key][0]
                else:
                    final_results[key] = np.array(final_results[key])
            self.save_results(final_results, game_name_str, agent_name)
            
  def tune_rates(self):
    lowest_multiplier=self.tuning_parameters['lowest_multiplier']
    highest_multiplier=self.tuning_parameters['highest_multiplier']
    size_grid_search=self.tuning_parameters['size_grid_search']
    log_step=(math.log(highest_multiplier)-math.log(lowest_multiplier))/(size_grid_search-1)
    base_constants=[lowest_multiplier*math.exp(i*log_step) for i in range(size_grid_search)]
    tuning_kwargs=self.training_kwargs.copy()
    tuning_kwargs['record_exploitabilities']=True
    tuning_kwargs['number_points']=None
    tuning_kwargs['log_interval']=self.global_init_kwargs['budget']
    tuning_kwargs['record_current']=False
    list_tasks = []
    lr_file_path = './configs/experiments/tuned_lrs_{}.yaml'.format(self.global_init_kwargs['budget'])
    if not os.path.exists(lr_file_path):
       tuned_lrs = {}
    else:
       tuned_lrs = yaml.load(open(lr_file_path, 'r'), Loader=yaml.FullLoader)
    for game_name in self.game_names:
      game_name_str = get_game_name_str(game_name)
      for agent_name in self.agent_names:
        if tuned_lrs.get(game_name_str) is None:
              tuned_lrs[game_name_str] = {}
        if tuned_lrs.get(game_name_str).get(agent_name) is None:
          ic('Tuning lr of agent={} on game={}'.format(agent_name, game_name))
          for base_constant in base_constants:
              list_tasks.append([
                  self.dict_agent_constructor[agent_name],
                  self.dict_agent_kwargs[agent_name],
                  game_name,
                  base_constant,
                  tuning_kwargs,
                  None,
                  0
                  ])
    if len(list_tasks)>0:
      ray.init()
      result_ids = []
      for task in list_tasks:
        result_ids.append(fit_agent.remote(*task))
      results = ray.get(result_ids)
      ray.shutdown()
      print("Finished tuning!")
    self.tuned_rates={}
    idx = 0
    for game_name in self.game_names:
        game_name_str = get_game_name_str(game_name)
        for agent_name in self.agent_names:
            if tuned_lrs.get(game_name_str).get(agent_name) is None:
              best_gap = math.inf
              for base_constant in base_constants:
                  gap = results[idx].get('average')[0]
                  if best_gap > gap:
                      best_gap = gap
                      tuned_lrs[game_name_str][agent_name]=base_constant
                  idx+=1
              ic('Add tuned lr={} of agent={} on game={}'.format(tuned_lrs[game_name_str][agent_name], agent_name, game_name))
    self.tuned_rates = tuned_lrs
    print("Best multipliers:")
    print(tuned_lrs)
    lr_file_path = './configs/experiments/tuned_lrs_{}.yaml'.format(self.global_init_kwargs['budget'])
    yaml_str = yaml.dump(self.tuned_rates)
    with open(lr_file_path, 'w') as f:
      f.write(yaml_str)

      
def get_game_name_str(game_name):
   game = pyspiel.load_game(game_name) if game_name != 'linear_game' else LinearGame()
   return '_'.join([game_name, 'H'+str(game.max_game_length()), 'A'+str(game.num_distinct_actions()), 'N'+str(game._noise_range)])
    

@ray.remote
def fit_agent(agent_contstructor, agent_kwargs, game_name, base_constant, training_kwargs, wandb_args, seed):
  setup_seed(seed)
  agent_kwargs['game'] =  pyspiel.load_game(game_name) if game_name != 'linear_game' else LinearGame()
  agent_kwargs['base_constant'] = base_constant
  agent = agent_contstructor(**agent_kwargs) 
  print(f'Train {agent.name} on {game_name}')
  ic(wandb_args)
  if wandb_args is not None and wandb_args['wandb_use']:
      job_type = ''
      if agent.name == 'F2TRL':
        job_type = '_'.join([agent.name, str(base_constant), str(agent_kwargs['use_cum_sta']), 'max({},{})'.format(agent_kwargs['scale_style'],str(agent_kwargs['scale_base'])),\
         'G{}'.format(str(agent_kwargs['ggamma']))])
      else:
        job_type = '_'.join([agent.name, str(base_constant)])
      wandb.init(project=wandb_args['wandb_project'],
        name=str(seed),
        group='_'.join([get_game_name_str(game_name), 'T'+str(agent_kwargs['budget']), 'N'+str(agent_kwargs['game']._noise_range)]),
        job_type = job_type,
        reinit=True,
        mode=wandb_args['mode'])
      if wandb_args['wandb_interval_log']:
        if game_name != 'linear_game':
          wandb.define_metric("explo_cur", step_metric='custom_step')
          wandb.define_metric("explo_avg", step_metric='custom_step')
        else:
          wandb.define_metric("custom_step")
          wandb.define_metric("cum_regret", step_metric='custom_step')
          wandb.define_metric("episode_regret", step_metric='custom_step')
      wandb.config.update(wandb_args)
      training_kwargs['wandb_obj'] = wandb
  else:
     training_kwargs['wandb_obj'] = None
  return agent.fit(**training_kwargs)


def fit_agent_wo_remote(agent_contstructor, agent_kwargs, game_name, base_constant, training_kwargs, wandb_args, seed):
  setup_seed(seed)
  agent_kwargs['game'] =  pyspiel.load_game(game_name) if game_name != 'linear_game' else LinearGame()
  agent_kwargs['base_constant'] = base_constant
  agent = agent_contstructor(**agent_kwargs) 
  print(f'Run {agent.name} on {game_name}')
  ic(wandb_args)
  if wandb_args is not None and wandb_args['wandb_use']:
      wandb.init(project=wandb_args['wandb_project'],
        name=str(seed),
        group='_'.join([get_game_name_str(game_name), 'T'+str(agent_kwargs['budget'])]),
        job_type='_'.join([agent.name, str(base_constant)]) if agent.name != 'F2TRL' \
          else '_'.join([agent.name, str(base_constant), str(agent_kwargs['use_cum_sta']), 'max({},{})'.format(agent_kwargs['scale_style'],str(agent_kwargs['scale_base']))]),
        reinit=True,
        mode=wandb_args['mode'])

      if wandb_args['wandb_interval_log']:
        if game_name != 'linear_game':
          wandb.define_metric("explo_cur", step_metric='custom_step')
          wandb.define_metric("explo_avg", step_metric='custom_step')
        else:
          wandb.define_metric("custom_step")
          wandb.define_metric("cum_regret", step_metric='custom_step')
          wandb.define_metric("episode_regret", step_metric='custom_step')
      wandb.config.update(wandb_args)
      training_kwargs['wandb_obj'] = wandb
  else:
     training_kwargs['wandb_obj'] = None
  return agent.fit(**training_kwargs)









