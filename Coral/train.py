import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import time


def main(device, peer):
    dataset_list = ['ASSIST', 'NIPS2020', 'junyi']
    dataset = dataset_list[0]

    # Load configuration
    with open('../data/' + dataset + '/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    epoch_n = 200
    lr = 0.05 #junyi
    lr = 0.02 #ASSIST
    beta = 0.5 #junyi
    beta = 0.25 #ASSIST
    peer_per = 1

    print(dataset)
    print(peer * peer_per)

    def train(ratio):
        data_loader = TrainDataLoader(dataset, ratio)
        net = Net(student_n, exer_n, knowledge_n, peer, peer_per, device)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        print('Training model...')

        loss_function = nn.NLLLoss()
        for epoch in range(epoch_n):
            data_loader.reset()
            running_loss = 0.0
            batch_count = 0
            status = 'train'
            while not data_loader.is_end():
                batch_count += 1
                print(batch_count)
                input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
                input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
                    device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
                optimizer.zero_grad()
                start_time = time.time()
                output_1, output_12, kl_loss, tc_loss = net.forward(status + str(batch_count), input_stu_ids,
                                                                    input_exer_ids, input_knowledge_embs, batch_count)
                end_time = time.time()  # End timing

                # Calculate the duration
                forward_duration = end_time - start_time
                print(f"Batch {batch_count} forward time: {forward_duration:.6f} seconds")
                output_0 = torch.ones(output_1.size()).to(device) - output_1
                output = torch.cat((output_0, output_1), 1)

                output_02 = torch.ones(output_12.size()).to(device) - output_12
                output2 = torch.cat((output_02, output_12), 1)

                loss = 0 * loss_function(torch.log(output + 1e-8), labels) + beta * (kl_loss + tc_loss)/2 + 1 * loss_function(
                    torch.log(output2 + 1e-8), labels) # junyi
                loss.backward()
                optimizer.step()
                net.apply_clipper()

                running_loss += loss.item()
                if batch_count % 200 == 199:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                    running_loss = 0.0

            # Validate and save current model every epoch
            save_snapshot(net, 'model/' + dataset + '/' + ratio + '/model_peer' + str(peer * peer_per) + '_beta_' + str(
                beta) + '_epoch' + str(epoch + 1))
            validate(net, epoch, ratio)

    def validate(model, epoch, ratio):
        data_loader = ValTestDataLoader('test', dataset, ratio)
        net = model
        print('Validating model...')
        data_loader.reset()
        net = net.to(device)
        net.eval()

        correct_count, exer_count = 0, 0
        batch_count, batch_avg_loss = 0, 0.0
        pred_all, label_all, binary_pre = [], [], []
        status = 'val'
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            output_1, output, kl_loss, tc_loss = net.forward(status, input_stu_ids, input_exer_ids,
                                                             input_knowledge_embs, batch_count)
            output = output.view(-1)

            for i in range(len(labels)):
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct_count += 1
                binary_pre.append(int(output[i] >= 0.5))
            exer_count += len(labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        accuracy = correct_count / exer_count
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        mae = np.mean(np.sqrt((label_all - pred_all) ** 2))
        auc = roc_auc_score(label_all, pred_all)
        binary_pre = np.array(binary_pre)
        f1 = f1_score(label_all, binary_pre)
        precision = precision_score(label_all, binary_pre)
        recall = recall_score(label_all, binary_pre)

        print(
            'dataset=%s, epoch= %d, accuracy= %f, rmse= %f, auc= %f, f1_score=%f, precision=%f, recall=%f, mae=%f\n' % (
            dataset + '-' + ratio, epoch + 1, accuracy, rmse, auc, f1, precision, recall, mae))

        with open('result/' + dataset + '/' + ratio + '/model_peer' + str(peer) + '_val.txt', 'a',
                  encoding='utf8') as f:
            f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, f1_score=%f, precision=%f, recall=%f, mae=%f\n' % (
            epoch + 1, accuracy, rmse, auc, f1, precision, recall, mae))

    def save_snapshot(model, filename):
        torch.save(model.state_dict(), filename)

    for ratio in [80]:
        train(str(ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified device and peer value.')
    parser.add_argument('--device', type=str, default='cuda:2', help='CUDA device to use (e.g., cuda:0, cuda:1, etc.)')
    parser.add_argument('--peer', type=int, default=80, help='Peer value to use for training')

    args = parser.parse_args()
    main(args.device, args.peer)
    # python train.py --device cuda:2 --peer 5
