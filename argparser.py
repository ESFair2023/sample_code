import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6, help="number of classes")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--data_path', default="./TrainingSet/")
    parser.add_argument('--save_path', default='./saved_model')
    parser.add_argument('--spd_para', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    return args
