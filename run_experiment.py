import yaml
import argparse
import wandb
from agents.utils import ExperimentGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/experiments/base_experiments.yaml', type=str, help='Path to configuration file.')
    parser.add_argument('--use_ray', default=1, type=int, help='Whether using ray.')
    parser.add_argument('--algo', default='F2TRL', type=str, help='Name of algo to be run.')
    parser.add_argument('--seed_idx', default=0, type=int, help='Index of the seeds.')
    parser.add_argument('--game', default='linear_game', type=str, help='Name of game to be test.')
    args = parser.parse_args()
    config_path = args.config
    exp_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)        
    exp_generator = ExperimentGenerator(args, **exp_config)
    tuning_parameters=exp_config.get('tuning_parameters')
    if tuning_parameters is not None and tuning_parameters.get('tune_parameters'):
        exp_generator.tune_rates()
    if args.use_ray:
        exp_generator.run()
    else:
        exp_generator.run_wo_ray()