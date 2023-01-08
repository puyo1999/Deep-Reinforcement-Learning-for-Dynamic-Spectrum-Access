import argparse

class Parser():
    def parse_args(args):
        parser = argparse.ArgumentParser(description='Training params')
        parser.add_argument('--type', type=str, default='DRQN', help="Algorithm to train from {A2C, DQN, DRQN}")
        parser.add_argument('--batch_size', type=int, default=6, help="Batch Size(experience replay)")
        parser.add_argument('--with_per', action='store_true', help='PER')
        parser.set_defaults()

        return parser.parse_args(args)
