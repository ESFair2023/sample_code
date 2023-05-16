from torch.utils.data import DataLoader
from data import MyDataset
import os
import torch
from argparser import args_parser
from experiment_log import PytorchExperimentLogger
from train import num_correct_pred

#testing
def test(test_loader, model, args):
    model.eval()
    test_num_data_per_class = [0] * args.num_classes
    test_num_correct_data_per_class = [0] * args.num_classes
    correct_predictions = 0
    total_targets = 0
    for i, (images, targets, G) in enumerate(test_loader):
        total_targets += targets.shape[0]
        unique_labels, counts = torch.unique(targets.cpu(), sorted=True, return_inverse=False, return_counts=True)
        for idx, label in enumerate(unique_labels):
            test_num_data_per_class[label] += counts[idx].item()
        images = images.to(args.device)
        targets = targets.to(args.device)
        output = model(images)
        _, pred = torch.max(output, dim=1)
        correct_predictions += torch.sum(pred == targets).item()
        for class_num in range(args.num_classes):
            res = num_correct_pred(output, targets, class_num)
            test_num_correct_data_per_class[class_num] += res

    overall_acc = correct_predictions / total_targets
    return overall_acc, correct_predictions, total_targets, G


if __name__ == '__main__':
    exp_logger = PytorchExperimentLogger('./log', "log", ShowTerminal=True)
    args = args_parser()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    path = args.data_path
    save_path = args.save_path
    groups = os.listdir(path)
    if '.DS_Store' in groups:
        groups.remove('.DS_Store')
    print(groups)
    test_sets = []

    groups_acc_dict = {}

    for i in range(len(groups)):
        print(path + groups[i])
        test_sets.append(
            DataLoader(MyDataset(path + groups[i], groups[i]), batch_size=64, shuffle=True, drop_last=True))


    #load model
    model = torch.load('./saved_model/model.pkl')
    model.to(args.device)

    correct_num = 0
    total_num = 0
    exp_logger.print("=" * 50 + '\n')

    for i, test_loader in enumerate(test_sets):
        acc, correct_num_per_group, total_num_per_group, G = test(test_loader, model, args)
        groups_acc_dict[G[0]] = acc
        correct_num += correct_num_per_group
        total_num += total_num_per_group
        exp_logger.print(f'Acc for Group {G[0]}: {acc}')

    overall_acc = correct_num / total_num

    print(f"Overall_Acc: {overall_acc}")
    exp_logger.print(f"Overall_Acc: {overall_acc}\n")
    abs_difference = 0
    minority_acc = groups_acc_dict['G10']
    for G, group_acc in groups_acc_dict.items():
        abs_difference += abs(group_acc - minority_acc)
    SPD = abs_difference / len(groups_acc_dict)
    fairness_score = ((args.spd_para - SPD) / args.spd_para) / 3
    acc_score = overall_acc / 3
    print(f"SPD: {SPD}")
    exp_logger.print(f"SPD: {SPD}\n")
    print(f"Fairness score: {fairness_score}")
    exp_logger.print(f"Fairness score: {fairness_score}\n")
    print(f"Accuracy score: {acc_score}")
    exp_logger.print(f"Accuracy score: {acc_score}\n")

    print(f"Performance score: {acc_score + fairness_score}")
    exp_logger.print(f"Performance score: {acc_score + fairness_score}\n")

