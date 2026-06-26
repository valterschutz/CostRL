import os
import sys
import warnings

os.environ['GYM_DISABLE_WARNINGS'] = '1'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='torch.triangular_solve is deprecated.*', category=UserWarning)

p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)

import argparse
from contextlib import redirect_stderr
from io import StringIO
import logging
from pathlib import Path
from pprint import pformat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    with redirect_stderr(StringIO()):
        from src.utils.config import read_config, namespace2dict

    hps = read_config(args.cfg_file)

    Path(hps.running.exp_dir).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(hps.running.exp_dir + f'/{args.mode}.log', mode='w')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[file_handler, console_handler],
        force=True,
    )
    logging.info('\n\n Config:\n\n'+pformat(namespace2dict(hps)))

    logging.info('Creating runner for agent=%s environment=%s', hps.agent.name, hps.environment.name)
    with redirect_stderr(StringIO()):
        from src.agents import get_runner
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
