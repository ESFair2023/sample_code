from torch.utils.data import DataLoader
from data import MyDataset
import os
import torch
import torch.nn as nn
import torchvision.models as models
from argparser import args_parser
from experiment_log import  PytorchExperimentLogger

#calculate the number of correct predictions
def num_correct_pred(output, target, class_num=None):
    if class_num != None:
        mask = (target == class_num)
        output = output[mask]
        target = target[mask]

    _, pred = torch.max(output, dim=1)
    correct_predictions = torch.sum(pred == target).item()

    return correct_predictions

#training
def train(train_loader, model, criterion, optimizer, args):
    num_data_per_class = [0] * args.num_classes #number of data in each class
    num_correct_data_per_class = [0] * args.num_classes #number of correct predictions in each class
    model.train()
    correct_predictions = 0
    total_targets = 0
    running_loss = 0.0
    for i, (images, targets, G) in enumerate(train_loader):
        total_targets += targets.shape[0]
        unique_labels, counts = torch.unique(targets.cpu(), sorted=True, return_inverse=False, return_counts=True)
        for idx, label in enumerate(unique_labels):
            num_data_per_class[label] += counts[idx].item()
        images = images.to(args.device)
        output = model(images)
        _, pred = torch.max(output, dim=1)
        targets = targets.to(args.device)
        loss = criterion(output, targets)
        running_loss += loss.item()
        correct_predictions += torch.sum(pred == targets).item()

        for class_num in range(args.num_classes):
            res = num_correct_pred(output, targets, class_num)
            num_correct_data_per_class[class_num] += res

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    overall_acc = correct_predictions / total_targets #calculate the overall accuracy
    return running_loss, overall_acc, correct_predictions, total_targets, G



if __name__ == '__main__':

    exp_logger = PytorchExperimentLogger('./log', "log", ShowTerminal=True)
    args = args_parser()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    path = args.data_path
    save_path = args.save_path
    groups = os.listdir(path)
    if '.DS_Store' in groups: #remove other files
        groups.remove('.DS_Store')
    print(groups)
    train_sets = []

    group_acc_dict = {}
    
    #put DataLoader with different groups
    for i in range(len(groups)):
        print(path + groups[i])
        train_sets.append(DataLoader(MyDataset(path + groups[i], groups[i]), batch_size = 64, shuffle=True, drop_last=True))


    #use pre-trained mobilenet_v2 as the sample
    num_classes = args.num_classes
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = 1280
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    groups_acc_dict = {}
    for iter in range(1, args.epochs + 1):
        correct_num = 0
        total_num = 0
        exp_logger.print("="*50+'\n')
        for i, train_loader in enumerate(train_sets):
            # get result after training
            loss, acc, correct_num_per_group, total_num_per_group, G = train(train_loader, model, loss_func, optimizer, args)

            groups_acc_dict[G[0]] = acc
            correct_num += correct_num_per_group
            total_num += total_num_per_group
            exp_logger.print(f'Epoch {iter}: running loss for Group {G[0]}: {loss}')
            exp_logger.print(f'Epoch {iter}: Acc for Group {G[0]}: {acc}')
        overall_acc = correct_num / total_num
        print(f"Epoch {iter}, Overall_Acc: {overall_acc}")
        exp_logger.print(f"Epoch {iter}, Overall_Acc: {overall_acc}\n")
        abs_difference = 0
        minority_acc = groups_acc_dict['G10'] #minority group: G10
        for G, group_acc in groups_acc_dict.items():
            abs_difference += abs(group_acc - minority_acc)
        SPD = abs_difference / len(groups_acc_dict)
        fairness_score = ((args.spd_para - SPD) / args.spd_para) / 3

        acc_score = overall_acc / 3
        print(f"Epoch {iter}, SPD: {SPD}")
        exp_logger.print(f"Epoch {iter}, SPD: {SPD}")
        print(f"Epoch {iter}, Fairness_score: {fairness_score}")
        exp_logger.print(f"Epoch {iter}, Fairness_score: {fairness_score}")
        print(f"Epoch {iter}, Accuracy score: {acc_score}")
        exp_logger.print(f"Epoch {iter}, Accuracy score: {acc_score}")

        print(f"Epoch {iter}, Performance score: {acc_score + fairness_score}")
        exp_logger.print(f"Epoch {iter}, Performance score: {acc_score + fairness_score}")


    torch.save(model, './saved_model/model.pkl')
    torch.save(model.state_dict(), './saved_model/model_state_dict.pkl')


