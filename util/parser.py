import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Training params')

    parser.add_argument('--type', type=str, default='DRQN', help="Algorithm to train from {A2C, DRQN}")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch Size(experience replay)")
