import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)

import argparse
import logging
from pathlib import Path
from pprint import pformat

from src.utils.config import read_config, dict2namespace, namespace2dict
from src.agents import get_runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    hps = read_config(args.cfg_file)

    Path(hps.running.exp_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(hps.running.exp_dir + f'/{args.mode}.log', mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info('\n\n Config:\n\n'+pformat(namespace2dict(hps)))

    logging.info('Creating runner for agent=%s environment=%s', hps.agent.name, hps.environment.name)
    runner = get_runner(hps)
    logging.info('Runner created')

    if args.mode == 'train':
        logging.info('Starting training')
        runner.train()
    elif args.mode == 'test':
        logging.info('Starting test')
        runner.test()
    elif args.mode == 'run':
        logging.info('Starting training')
        runner.train()
        logging.info('Starting test')
        runner.test()
    else:
        raise ValueError()

if __name__ == "__main__":
    main()
