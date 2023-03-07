import argparse

class Parser():
    def parse_args(args):
        parser = argparse.ArgumentParser(description='Training params')
        parser.add_argument('--time_slot', type=int, default=10000, help="Time Slot")
        parser.add_argument('--type', type=str, default='DRQN', help="Algorithm to train from {A2C, DQN, DRQN}")
        parser.add_argument('--batch_size', type=int, default=6, help="Batch Size(experience replay)")
        parser.add_argument('--gamma', type=float, default=0.99, help="Discount Rate of future rewards")
        parser.add_argument('--hidden', type=int, default=128, help="Hidden Neuron count")
        parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
        parser.add_argument('--explore_start', type=float, default=.02, help="탐험 시작")
        parser.add_argument('--explore_stop', type=float, default=.01, help="탐험 종료")
        parser.add_argument('--decay_rate', type=float, default=.0001, help="탐험 감소율")
        parser.add_argument('--with_per', action='store_true', help='PER')
        parser.set_defaults()

        return parser.parse_args(args)
