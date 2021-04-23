import argparse
import numpy as np
import time
from numpy import load
from numpy import save
import os
import torch
import torch.nn as nn

from model import DeepGRU
from model import TCN
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log                  # Logging

import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    help='dataset to train on: ' + ' | '.join(DataFactory.dataset_names),
                    default='sbu')
parser.add_argument('--seed', type=int, metavar='N',
                    help='random number generator seed, use "-1" for random seed',
                    default=1570254494)
parser.add_argument('--num-synth', type=int, metavar='N',
                    help='number of synthetic samples to generate',
                    default=1)
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)


# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
seed = int(time.time()) if args.seed == -1 else args.seed
use_cuda = torch.cuda.is_available() and args.use_cuda


# ----------------------------------------------------------------------------------------------------------------------
def main():
    
    """
    adv_data = load('data.npy')
    adv_label = []
    for i in range(46):
        if i < 27:
            adv_label.append(torch.from_numpy(adv_data[i+1]))
        else:
            adv_label.append(torch.from_numpy(adv_data[-1]))
    adv_label = torch.stack(adv_label)
    """

    # Load the dataset
    log.set_dataset_name(args.dataset)
    dataset = DataFactory.instantiate(args.dataset, args.num_synth)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)
    # print("dataset: ", dataset)
    # Run each fold and average the results
    losses = []
    losses2 = []
    #for fold_idx in range(dataset.num_folds):
    for fold_idx in range(1):
        log('Running fold "{}"...'.format(fold_idx))

        test_loss, test_loss2 = run_fold(dataset, fold_idx, use_cuda)
        losses += [test_loss]
        losses2 += [test_loss2]

        log('Fold "{}" complete, final loss 1: {}'.format(fold_idx, test_loss))
        log('Fold "{}" complete, final loss 2: {}'.format(fold_idx, test_loss2))
    log('')
    log('-----------------------------------------------------------------------')
    log('Training complete!')
    log('Average losses 1: {}'.format(np.mean(losses)))
    log('Average losses 2: {}'.format(np.mean(losses2)))

# ----------------------------------------------------------------------------------------------------------------------
def run_fold(dataset, fold_idx, use_cuda):
    """
    Trains/tests the model on the given fold
    """

    hyperparameters = dataset.get_hyperparameter_set()
    # Instantiate the model, loss measure and optimizer
    #model = DeepGRU(dataset.num_features, dataset.num_classes)
    #channel_sizes = [256] * 10
    #model2 = TCN(45, 45, channel_sizes, 2, 0.0)
    model = torch.load("rnn5.pt")
    model2 = torch.load("cnn5.pt")
    #model = torch.load("0_rnn_adv_max_beta_50000_epoch99.pt")
    #model2 = torch.load("0_cnn_adv_max_beta_50000_epoch99.pt")
    #criterion = nn.MSELoss()
    criterion = multivariate_MSE_loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyperparameters.learning_rate,
                                 weight_decay=hyperparameters.weight_decay)

    optimizer2 = torch.optim.Adam(model2.parameters(),
                                 lr=hyperparameters.learning_rate,
                                 weight_decay=hyperparameters.weight_decay)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        model2 = torch.nn.DataParallel(model2).cuda()

    # Create data loaders
    train_loader, test_loader = dataset.get_data_loaders(fold_idx,
                                                         shuffle=False,
                                                         random_seed=seed+fold_idx,
                                                         normalize=False)

    best_train_loss = 9999
    best_test_loss = 9999
    best_train_loss2 = 9999
    best_test_loss2 = 9999
    
    # Train the model
    #for epoch in range(hyperparameters.num_epochs):
    for epoch in range(100):
        """        
        loss_meter = AverageMeter()
        train_meter = AverageMeter()
        test_meter = AverageMeter()
        
        loss_meter2 = AverageMeter()
        train_meter2 = AverageMeter()
        test_meter2 = AverageMeter()
        
        #
        # Training loop
        #
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            curr_batch_size, loss = run_batch(batch, model, criterion)
            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update stats
            loss_meter.update(loss.item(), curr_batch_size)
            #train_meter.update(accuracy, curr_batch_size)
                        
            model2.train()
            optimizer2.zero_grad()

            curr_batch_size2, loss2 = run_batch(batch, model2, criterion)
            # Backward and optimize
            loss2.backward()
            optimizer2.step()

            # Update stats
            loss_meter2.update(loss2.item(), curr_batch_size2)
            #print("loss 1: ", loss, "loss 2: ", loss2)
            
        train_loss = loss_meter.avg
        train_loss2 = loss_meter2.avg

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if train_loss2 < best_train_loss2:
            best_train_loss2 = train_loss2

        log('Epoch: [{0}]'.format(epoch))
        log('       [Avg Train Loss RNN]          {loss.avg:.6f}'.format(loss=loss_meter))
        log('       [Avg Train Loss CNN]          {loss2.avg:.6f}'.format(loss2=loss_meter2))
        #log('       [Training]   Prec@1 {top1.avg:.6f} Max {max:.6f}'
             #.format(top1=train_meter, max=best_train_accuracy))

        test_loss_meter = AverageMeter()
        test_loss_meter2 = AverageMeter()

        for batch in test_loader:
            model.train()
            optimizer.zero_grad()

            curr_batch_size, loss = run_batch(batch, model, criterion)
            loss.backward()
            optimizer.step()
            test_loss_meter.update(loss.item(), curr_batch_size)
            #test_meter.update(accuracy, curr_batch_size)
            

            model2.train()
            optimizer2.zero_grad()    
            curr_batch_size2, loss2 = run_batch(batch, model2, criterion)
            loss2.backward()
            optimizer2.step()
            test_loss_meter2.update(loss2.item(), curr_batch_size2)
                
        test_loss = test_loss_meter.avg
        test_loss2 = test_loss_meter2.avg

        # Update best accuracies
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            #with open("rnn5.pt", 'wb') as f1:
            #    print('Save model!\n')
            #    torch.save(model, f1)
            
        if best_test_loss2 > test_loss2:
            best_test_loss2 = test_loss2
            #with open("cnn5.pt", 'wb') as f2:
            #    print('Save model!\n')
            #    torch.save(model2, f2)            

        if epoch % 10 == 9:
            #torch.save(model.state_dict(), os.path.join('adv_training', 'rnn_adv_beta_1_max_epoch{}.pt'.format(epoch)))
            with open("0_rnn_adv_max_beta_1_epoch{}.pt".format(epoch), 'wb') as f1:
                print('Save model RNN!\n')
                torch.save(model, f1)
            #torch.save(optimizer.state_dict(), os.path.join(''rnn_adv_beta_1_max_checkpoint_epoch{}.tar'.format(epoch)))
            with open("0_cnn_adv_max_beta_1_epoch{}.pt".format(epoch), 'wb') as f2:
                print('Save model CNN!\n')
                torch.save(model2, f2)
            #torch.save(model2.state_dict(), os.path.join('adv_training', 'cnn_adv_beta_1_max_epoch{}.pt'.format(epoch)))
            #torch.save(optimizer2.state_dict(), os.path.join('adv_training', 'cnn_adv_beta_1_max_checkpoint_epoch{}.tar'.format(epoch)))


        log('       [Avg Test Loss RNN]          {loss.avg:.6f}'.format(loss=test_loss_meter))
        log('       [Avg Test Loss CNN]          {loss2.avg:.6f}'.format(loss2=test_loss_meter2))
        #log('       [Validation] Prec@1 {top1:.6f} Max {max:.6f}'
            #.format(top1=test_accuracy, max=best_test_accuracy))

        #
        # Testing loop
        #
        """ 
        
        """
        model.train()
        model2.train()
        with torch.no_grad():
            test_loss_meter = AverageMeter()
            test_loss_meter2 = AverageMeter()

            for batch in test_loader:

                curr_batch_size, loss = run_batch(batch, model, criterion)
                test_loss_meter.update(loss.item(), curr_batch_size)
                #test_meter.update(accuracy, curr_batch_size)
                
                curr_batch_size2, loss2 = run_batch(batch, model2, criterion)
                test_loss_meter2.update(loss2.item(), curr_batch_size2)
                
            test_loss = test_loss_meter.avg
            test_loss2 = test_loss_meter2.avg

            # Update best accuracies
            if best_test_loss > test_loss:
                best_test_loss = test_loss
                #with open("rnn2.pt", 'wb') as f1:
                #    print('Save model!\n')
                #    torch.save(model, f1)
            
            if best_test_loss2 > test_loss2:
                best_test_loss2 = test_loss2
                #with open("cnn2.pt", 'wb') as f2:
                #    print('Save model!\n')
                #    torch.save(model2, f2)            

            #if epoch % 25 == 0:
            #    with open("rnn_adv_{}.pt".format(epoch), 'wb') as f1:
            #        print('Save model RNN!\n')
            #        torch.save(model, f1)

            #    with open("cnn_adv_{}.pt".format(epoch), 'wb') as f2:
            #        print('Save model CNN!\n')
            #        torch.save(model2, f2)


            log('       [Avg Test Loss RNN]          {loss.avg:.6f}'.format(loss=test_loss_meter))
            log('       [Avg Test Loss CNN]          {loss2.avg:.6f}'.format(loss2=test_loss_meter2))
            #log('       [Validation] Prec@1 {top1:.6f} Max {max:.6f}'
                 #.format(top1=test_accuracy, max=best_test_accuracy))
        
        #if loss_meter.avg <= 1e-6: #or best_test_accuracy == 100:
        #    break
        """
        
        if epoch == 99:
            for batch in test_loader:
                #model.train()
                #optimizer.zero_grad()

                #curr_batch_size, loss = run_batch(batch, model, criterion)
                #loss.backward()
                #optimizer.step()
                #test_loss_meter.update(loss.item(), curr_batch_size)
            #test_meter.update(accuracy, curr_batch_size)
            

                #model2.train()
                #optimizer2.zero_grad()    
                #curr_batch_size2, loss2 = run_batch(batch, model2, criterion)
                #loss2.backward()
                #optimizer2.step()
                #test_loss_meter2.update(loss2.item(), curr_batch_size2)

                model = torch.load("0_rnn_adv_max_beta_1_random_xy_epoch99.pt")
                model2 = torch.load("0_cnn_adv_max_beta_1_random_xy_epoch99.pt")
                #model = torch.load("rnn5.pt")
                #model2 = torch.load("cnn5.pt")
                model.train()
                model2.train()
                counter_natural, counter_robust = run_batch_adv_test(batch, model, model2, criterion)
                counter_natural_2, counter_robust_2 = run_batch_adv_test(batch, model2, model, criterion)
                print("RNN: ", counter_natural/55, counter_robust/55)
                print("CNN: ", counter_natural_2/55, counter_robust_2/55)
            #eval_adv_test_whitebox(model, model2, "cuda", test_loader)
        
            #eval_adv_test_whitebox(model2, "cuda", train_loader)
    return best_test_loss, best_test_loss2


# ----------------------------------------------------------------------------------------------------------------------
def run_batch(batch, model, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch
    new_examples = []
    new_labels = []
    new_lengths = []
    for i in range(len(examples)):
        curr_seq = []
        curr_label = []
        for j in range(len(examples[i])):
            if (j % 2) == 0:
                curr_seq.append(examples[i][j])
            else:
                curr_label.append(examples[i][j])
        new_examples.append(torch.stack(curr_seq))
        new_labels.append(torch.stack(curr_label))
        new_lengths.append(lengths[i]/2)
    
    for i in range(len(examples)):
        curr_seq = []
        curr_label = []
        for j in range(len(examples[i])):
            if (j % 2) == 1:
                curr_seq.append(examples[i][j])
            else:
                curr_label.append(examples[i][j])
        new_examples.append(torch.stack(curr_seq))
        new_labels.append(torch.stack(curr_label))
        new_lengths.append(lengths[i]/2)

    new_examples = torch.stack(new_examples)
    new_labels = torch.stack(new_labels)
    new_lengths = torch.stack(new_lengths)
    
    if use_cuda:
        new_examples = new_examples.cuda()
        new_labels = new_labels.cuda()
        new_lengths = new_lengths.cuda()

    # Forward and loss computation

    """ GRU Model """
    outputs = model(new_examples)

    """ TCN Model """
    #outputs = model(new_examples)
    loss = criterion(outputs, new_labels)
    #loss2 = 0
    loss2 = standard(model, new_examples, new_labels)
    #print("L-natural: ", loss, ", L-adv: ", loss2) 
    # Compute the accuracy
    #predicted = outputs.argmax(1)
    #correct = (predicted == labels).sum().item()
    curr_batch_size = new_labels.size(0)
    #accuracy = correct / curr_batch_size * 100.0

    beta = 1
    return curr_batch_size, (loss + beta * loss2)
    #return accuracy, curr_batch_size, loss

def run_batch_adv_test(batch, model, model2, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch
    new_examples = []
    new_labels = []
    new_lengths = []
    old_labels = []
    for i in range(len(examples)):
        curr_seq = []
        curr_label = []
        for j in range(len(examples[i])):
            if (j % 2) == 0:
                curr_seq.append(examples[i][j])
            else:
                curr_label.append(examples[i][j])
        new_examples.append(torch.stack(curr_seq))
        new_labels.append(torch.stack(curr_label))
        new_lengths.append(lengths[i]/2)
        old_labels.append(labels[i])


    new_examples_2 = []
    new_labels_2 = []
    new_lengths_2 = []
    old_labels_2 = []
    for i in range(len(examples)):
        curr_seq = []
        curr_label = []
        for j in range(len(examples[i])):
            if (j % 2) == 1:
                curr_seq.append(examples[i][j])
            else:
                curr_label.append(examples[i][j])
        new_examples_2.append(torch.stack(curr_seq))
        new_labels_2.append(torch.stack(curr_label))
        new_lengths_2.append(lengths[i]/2)
        old_labels_2.append(labels[i])


    new_examples = torch.stack(new_examples)
    new_labels = torch.stack(new_labels)
    new_lengths = torch.stack(new_lengths)
    new_examples_2 = torch.stack(new_examples_2)
    new_labels_2 = torch.stack(new_labels_2)
    new_lengths_2 = torch.stack(new_lengths_2)
    
    if use_cuda:
        new_examples = new_examples.cuda()
        new_labels = new_labels.cuda()
        new_lengths = new_lengths.cuda()
        new_examples_2 = new_examples_2.cuda()
        new_labels_2 = new_labels_2.cuda()
        new_lengths_2 = new_lengths_2.cuda()
    # Forward and loss computation
    counter_natural = 0
    counter_robust = 0
    """ GRU Model """
    #for i in range(len(new_examples)):
    #    output = model(new_examples[i])
    #    natural_loss = criterion(outputs, new_labels)
    #    if natural_loss > 60:
    #         print("nat: ", natural_loss, i)
    #    robust_loss = standard(model, new_examples[i], new_labels[i])
    #    if robust_loss > 93.17:
    #         print("rob: ", robust_loss, i)
    #outputs = model(new_examples)

    """ TCN Model """
    outputs = model(new_examples)
    X_pgd = standard(model, new_examples, new_labels)
    adv_outputs = model(X_pgd)
    outputs_2 = model(new_examples_2)
    X_pgd_2 = standard(model, new_examples_2, new_labels_2)
    adv_outputs_2 = model(X_pgd_2)
    
    for i in range(len(new_examples)):
        natural_loss = criterion(torch.tensor(outputs[i]), torch.tensor(new_labels[i]))
        robust_loss = criterion(torch.tensor(adv_outputs[i]), torch.tensor(new_labels[i]))
        natural_loss_2 = criterion(torch.tensor(outputs_2[i]), torch.tensor(new_labels_2[i]))
        robust_loss_2 = criterion(torch.tensor(adv_outputs_2[i]), torch.tensor(new_labels_2[i]))  
        print(old_labels[i].item(), old_labels_2[i].item(), robust_loss, robust_loss_2) 
        
        if old_labels[i].item() == 7:
            if natural_loss > 84.28 and natural_loss_2 > 84.28:
                #counter_natural = counter_natural + 1
                counter_robust = counter_robust + 1
                #print(i)
            elif natural_loss > 84.28 and natural_loss_2 <= 84.28:
                counter_natural = counter_natural + 1
                if robust_loss_2 > 84.28:
                    counter_robust = counter_robust + 1
                #print(i)
            elif natural_loss <= 84.28 and natural_loss_2 > 84.28:
                counter_natural = counter_natural + 1
                if robust_loss > 84.28:
                    counter_robust = counter_robust + 1  
                #print(i)          
            else:
                counter_natural = counter_natural + 1
                if robust_loss > 84.28 or robust_loss_2 > 84.28:
                    counter_robust = counter_robust + 1
        
        elif old_labels[i].item() == 4:
            if natural_loss > 100.61 and natural_loss_2 > 100.61:
                #counter_natural = counter_natural + 1
                counter_robust = counter_robust + 1
            elif natural_loss > 100.61 and natural_loss_2 <= 100.61:
                counter_natural = counter_natural + 1
                if robust_loss_2 > 100.61:
                    counter_robust = counter_robust + 1
            elif natural_loss <= 100.61 and natural_loss_2 > 100.61:
                counter_natural = counter_natural + 1
                if robust_loss > 100.61:
                    counter_robust = counter_robust + 1
            else:
                counter_natural = counter_natural + 1
                if robust_loss > 100.61 or robust_loss_2 > 100.61:
                    counter_robust = counter_robust + 1

        elif old_labels[i].item() == 1:
            if natural_loss > 76.78 and natural_loss_2 > 76.78:
                #counter_natural = counter_natural + 1
                counter_robust = counter_robust + 1
            elif natural_loss > 76.78 and natural_loss_2 <= 76.78:
                counter_natural = counter_natural + 1
                if robust_loss_2 > 76.78:
                    counter_robust = counter_robust + 1
            elif natural_loss <= 76.78 and natural_loss_2 > 76.78:
                counter_natural = counter_natural + 1
                if robust_loss > 76.78:
                    counter_robust = counter_robust + 1
            else:
                counter_natural = counter_natural + 1
                if robust_loss > 76.78 or robust_loss_2 > 76.78:
                    counter_robust = counter_robust + 1
        
        elif old_labels[i].item() == 6:
            if natural_loss > 23.98 and natural_loss_2 > 23.98:
                #counter_natural = counter_natural + 1
                counter_robust = counter_robust + 1
            elif natural_loss > 23.98 and natural_loss_2 <= 23.98:
                counter_natural = counter_natural + 1
                if robust_loss_2 > 23.98:
                    counter_robust = counter_robust + 1
            elif natural_loss <= 23.98 and natural_loss_2 > 23.98:
                counter_natural = counter_natural + 1
                if robust_loss > 23.98:
                    counter_robust = counter_robust + 1
            else:
                counter_natural = counter_natural + 1
                if robust_loss > 23.98 or robust_loss_2 > 23.98:
                    counter_robust = counter_robust + 1

        elif old_labels[i].item() == 2:
            if natural_loss > 46.25 and natural_loss_2 > 46.25:
                #counter_natural = counter_natural + 1
                counter_robust = counter_robust + 1
            elif natural_loss > 46.25 and natural_loss_2 <= 46.25:
                counter_natural = counter_natural + 1
                if robust_loss_2 > 46.25:
                    counter_robust = counter_robust + 1
            elif natural_loss <= 46.25 and natural_loss_2 > 46.25:
                counter_natural = counter_natural + 1
                if robust_loss > 46.25:
                    counter_robust = counter_robust + 1
            else:
                counter_natural = counter_natural + 1
                if robust_loss > 46.25 or robust_loss_2 > 46.25:
                    counter_robust = counter_robust + 1
        
        else:
            if natural_loss > 66.38 and natural_loss_2 > 66.38:
                #counter_natural = counter_natural + 1
                counter_robust = counter_robust + 1
            elif natural_loss > 66.38 and natural_loss_2 <= 66.38:
                counter_natural = counter_natural + 1
                if robust_loss_2 > 66.38:
                    counter_robust = counter_robust + 1
            elif natural_loss <= 66.38 and natural_loss_2 > 66.38:
                counter_natural = counter_natural + 1
                if robust_loss > 66.38:
                    counter_robust = counter_robust + 1
            else:
                counter_natural = counter_natural + 1
                if robust_loss > 66.38 or robust_loss_2 > 66.38:
                    counter_robust = counter_robust + 1
            
        #print(counter_robust) 
        
        #print("RNN: ", natural_loss, robust_loss)
        #print("CNN: ", natural_loss_2, robust_loss_2)
    #print(a, b)
    #loss2 = 0
    #loss2 = standard(model, new_examples, new_labels)
    #print("L-natural: ", loss, ", L-adv: ", loss2) 
    # Compute the accuracy
    #predicted = outputs.argmax(1)
    #correct = (predicted == labels).sum().item()
    #curr_batch_size = new_labels.size(0)
    #accuracy = correct / curr_batch_size * 100.0

    #beta = 0.1
    #return curr_batch_size, (loss + beta * loss2)
    #return accuracy, curr_batch_size, loss
    return counter_natural, counter_robust

def standard(model,
                X,
                y,
                step_size=0.03,
                epsilon=0.45,
                num_steps=20,
                lbda=0.1):

    out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    #if args.random:
    #random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to("cuda")
    #d = torch.tensor([0.0, 0.0, 1.0])
    #r6 = d.repeat(X_pgd.shape[0], X_pgd.shape[1], 15).to("cuda")
    #X_pgd = Variable(X_pgd.data + random_noise * r6, requires_grad=True)
    #y = y.data + random_noise * r6
    model.train()
    for _ in range(num_steps):
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = multivariate_untargeted_loss(model(X_pgd), y, y, X_pgd, 100.61, lbda)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        c = torch.tensor([0.0, 0.0, 1.0])
        r5 = c.repeat(X_pgd.shape[0], X_pgd.shape[1], 15).to("cuda")
        #r5 = c.repeat(X_pgd.shape[0], 15).to("cuda")
        #print(_)
        #print(eta.shape)
        #r5 = c.repeat(X_pgd.shape[0], 15).to("cuda")
        eta = eta * r5
        X_pgd = Variable(X.data + eta, requires_grad=True)
    # zero gradient
    opt.zero_grad()
    # calculate min loss
    loss = multivariate_MSE_loss(model(X_pgd), y)
    #F.cross_entropy(model(x_adv), y)
    #return loss
    return X_pgd

def multivariate_MSE_loss(output, target):
    loss = Variable(torch.zeros(1)).cuda()
    for i in range(len(target)):
        for j in range(len(target[i])):
            loss = loss + torch.norm((target[i][j] - output[i][j]), 2)
    return loss

def multivariate_regression_loss(output, target, orig, X_pgd, kappa, lbda = 0.1):
    loss_spatial = Variable(torch.zeros(1)).cuda()
    loss_temporal = Variable(torch.zeros(1)).cuda()
    e = torch.tensor([1.0, 1.0, 0.0])
    r5 = e.repeat(output.shape[0], output.shape[1], 15).to("cuda")
    output = output * r5
    target = target * r5
    orig = orig * r5
    for i in range(len(target)):
        #loss1 = loss1 + torch.norm(target[i] - output[i], float("inf"))
        for j in range(len(target[i])):
            if kappa > torch.norm((target[i][j] - output[i][j]), 2):
                loss_spatial = loss_spatial + (torch.norm((target[i][j] - output[i][j]), 2) - kappa / len(target[i]))
            else:
                loss_spatial = loss_spatial + (torch.norm((target[i][j] - output[i][j]), 2) + kappa / len(target[i]))
            if j != 0:
                loss_temporal = loss_temporal + torch.norm((X_pgd[i][j] - X_pgd[i][j-1]),2)
            
    loss = -loss_spatial - lbda * loss_temporal
    return loss

def multivariate_untargeted_loss(output, target, orig, X_pgd, kappa, lbda = 0.1):
    loss_spatial = Variable(torch.zeros(1)).cuda()
    loss_temporal = Variable(torch.zeros(1)).cuda()
    e = torch.tensor([1.0, 1.0, 0.0])
    r5 = e.repeat(output.shape[0], output.shape[1], 15).to("cuda")
    #r5 = e.repeat(output.shape[0], 15).to("cuda")
    output = output * r5
    target = target * r5
    orig = orig * r5
    for i in range(len(target)):
        #loss1 = loss1 + torch.norm(target[i] - output[i], float("inf"))
        for j in range(len(target[i])):
            if (kappa / len(target[i])) > torch.norm((target[i][j] - output[i][j]), 2):
                loss_spatial = loss_spatial + (-torch.norm((target[i][j] - output[i][j]), 2) + kappa / len(target[i]))
            else:
                loss_spatial = loss_spatial #- 1/torch.norm((target[i][j] - output[i][j]), 2)
            if j != 0:
                loss_temporal = loss_temporal + torch.norm((X_pgd[i][j] - X_pgd[i][j-1]),2)

    loss = -loss_spatial - lbda * loss_temporal
    return loss

def multivariate_regression_loss_2(output, target, orig, X_pgd):
    loss1 = Variable(torch.zeros(1)).cuda()
    loss2 = Variable(torch.zeros(1)).cuda()
    e = torch.tensor([1.0, 1.0, 0.0])
    r5 = e.repeat(output.shape[0], output.shape[1], 15).to("cuda")
    output = output * r5
    target = target * r5
    orig = orig * r5
    for i in range(len(target)):
        #loss1 = loss1 + torch.norm(target[i] - output[i], float("inf"))
        for j in range(len(target[i])):
            loss2 = loss2 + torch.norm((target[i][j] - output[i][j]), 2)
            if j != 0:
                loss1 = loss1 + torch.norm((X_pgd[i][j] - X_pgd[i][j-1]),2)

    loss = -loss2
    return loss

def multivariate_linf_loss(output, target, orig):
    loss1 = Variable(torch.zeros(1)).cuda()
    loss2 = Variable(torch.zeros(1)).cuda()
    e = torch.tensor([1.0, 1.0, 0.0])
    r5 = e.repeat(output.shape[0], output.shape[1], 15).to("cuda")
    output = output * r5
    target = target * r5
    orig = orig * r5
    for i in range(len(target)):
        #loss1 = loss1 + torch.norm(target[i] - output[i], float("inf"))
        for j in range(len(target[i])):
            loss2 = loss2 + torch.norm((target[i][j] - output[i][j]), float('inf'))
    loss = -loss2
    return loss

def _pgd_whitebox(model,
                  X,
                  lengths,
                  y,
                  orig,
                  model_name,
                  model2,
                  epsilon=0.45,
                  i=0,
                  num_steps=400,
                  step_size=0.03,
                  kappa=65,
                  lbda=0.1):

    out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    #if args.random:
    #random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to("cuda")
    #X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            #loss = multivariate_regression_loss(model(X_pgd, lengths), y, orig)
            loss = multivariate_regression_loss(model(X_pgd), y, orig, X_pgd, kappa, lbda)
            #loss = multivariate_linf_loss(model(X_pgd), y, orig)
        loss.backward()
         
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        c = torch.tensor([0.0, 0.0, 1.0])
        r5 = c.repeat(X_pgd.shape[0], X_pgd.shape[1], 15).to("cuda")
        #r5 = c.repeat(X_pgd.shape[0], 15).to("cuda")
        eta = eta * r5
        X_pgd = Variable(X.data + eta, requires_grad=True)
          
        if _ == 399: 
            print("-------------------------------------------------")
            print(i, " Whitebox Loss: ", epsilon, model_name, _, multivariate_regression_loss_2(model(X_pgd), y, orig, X_pgd).item())

    orig_out = model(X)
    adv_out = model(X_pgd)
    orig_seq = torch.cat((X, orig_out), 2)
    adv_seq = torch.cat((X_pgd, adv_out), 2)
    gt_seq = torch.cat((X, orig), 2)
    gt_adv_seq = torch.cat((X_pgd, y), 2)
    
    #print("Blackbox Loss: ", epsilon, multivariate_regression_loss(model2(X_pgd), y, orig, X_pgd))
    
    #return multivariate_regression_loss_2(model(X_pgd), y, orig, X_pgd), multivariate_regression_loss_2(model2(X_pgd), y, orig, X_pgd)
    
    for l in range(len(X)):
        #print(lengths[i].item())
        fname = model_name + "_sample_2_"+ str(i) + "_"  + str(epsilon) + "_adv.npy"
        save(fname, adv_seq.detach().cpu().numpy()[0][0:lengths[l].item()])
        fname2 = model_name +"_sample_2_"+ str(i) + "_" + str(epsilon) + "_orig.npy"
        save(fname2, orig_seq.detach().cpu().numpy()[0][0:lengths[l].item()])
        fname3 = model_name + "_sample_2_" + str(i)  +  "_gt.npy"
        save(fname3, gt_seq[l].detach().cpu().numpy()[0:lengths[l].item()])
        fname4 = model_name + "_sample_2_" + str(i) + "_x.npy"
        save(fname4, gt_adv_seq[l].detach().cpu().numpy()[0:lengths[l].item()])
     
def eval_adv_test_whitebox(model, model2, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.train()
    robust_err_total = 0
    natural_err_total = 0
    # 0 1 2 3 4 6
    adv_data = load('data_sam.npy')
    # 5 7
    adv_data2 = load('data2_sam.npy')
    adv_label = []
    adv_label2 = []
    #print(len(adv_data))
    for i in range(12, 58):
        if (2 * i) < len(adv_data):
            # Example Set 2 3 4
            adv_label.append(torch.from_numpy(adv_data[2*i]))

            # Example Set 1 5
            adv_label2.append(torch.from_numpy(adv_data[2*i+1]))
        else:
            # Example Set 2 3 4
            adv_label.append(torch.from_numpy(adv_data[-2]))

            # Example Set 1 5
            adv_label2.append(torch.from_numpy(adv_data[-1]))

    for data, lengths, target in test_loader:
        #print(target)
        new_examples = []
        new_labels = []
        new_examples2 = []
        new_labels2 = []
        new_lengths = []
        original_labels = []
        original_labels2 = []
        adversarial_label = []
        rnnwbs = {}
        cnnwbs = {}
        rnnbbs = {}
        cnnbbs = {}
        w = 3
        #for w in range(8):
        if w == 3:  
            for i in range(len(data)):
                curr_seq = []
                curr_label = []
                curr_seq2 = []
                curr_label2 = []
                #adv_label2 = []
                # Example Set 1
                #if target[i].item() == 7:
                # Example Set 3
                #if target[i].item() == 0:
                # Example Set 5
                #if target[i].item() == 3:
                #print(target[i])
                # Kicking
                if target[i].item() == w:
                    #print(1)
                    for j in range(len(data[i])):
                        # Example Set 1 3 4 5

                       
                        if (j % 2) == 1:

                        # Example Set 2
                        
                        #if (j % 2) == 0:
                            curr_seq.append(data[i][j])
                            curr_label2.append(data[i][j])
                        else:
                            # Example Set 1 2 4
                            curr_label.append(data[i][j])
                            curr_seq2.append(data[i][j])
                            # Example Set 3
                            #adv_label2.append(data[i][0])
                    #print(curr_seq)
                    new_examples.append(torch.stack(curr_seq))
                    # Example Set 1 2 4
                    new_labels.append(torch.stack(adv_label))
                    # Example Set 3
                    #new_labels.append(torch.stack(adv_label2))
                    original_labels.append(torch.stack(curr_label))
                    new_lengths.append(lengths[i]/2)

                    new_examples2.append(torch.stack(curr_seq2))
                    # Example Set 1 2 4
                    new_labels2.append(torch.stack(adv_label2))
                    # Example Set 3
                    #new_labels.append(torch.stack(adv_label2))
                    original_labels2.append(torch.stack(curr_label2))
            
            # Set 1/3/4: 1, Set 1: 3
            new_examples_ = torch.stack([new_examples[1]])
            new_labels_ = torch.stack([new_labels[1]])
            new_lengths_ = torch.stack([new_lengths[1]])
            original_labels_ = torch.stack([original_labels[1]])
            if use_cuda:
                new_examples_ = new_examples_.cuda()
                new_labels_ = new_labels_.cuda()
                new_lengths_ = new_lengths_.cuda()
                original_labels_ = original_labels_.cuda()
            data_, lengths_, target_, orig = new_examples_.to(device), new_lengths_.to(device), new_labels_.to(device), original_labels_.to(device)
            #data2_, lengths_, target2_, orig2 = new_examples_2.to(device), new_lengths_.to(device), new_labels_2.to(device), original_labels_2.to(device)
            X, y, z = Variable(data_, requires_grad=True), Variable(target_), Variable(orig)
            #_pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.075, 1)
            _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.075, 3)
            #_pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.150, 1)
            _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.150, 3)           
            #_pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.225, 1)
            _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.225, 3)
            #_pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.300, 1)
            _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.300, 3)
            #_pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.375, 1)
            _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.375, 3)
            #_pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.450, 1)
            _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.450, 3)
            
            """
            rnnwb = {'0075': [], '0150': [], '0225': [], '0300': [], '0375': [], '0450': []}
            cnnwb = {'0075': [], '0150': [], '0225': [], '0300': [], '0375': [], '0450': []}
            rnnbb = {'0075': [], '0150': [], '0225': [], '0300': [], '0375': [], '0450': []}
            cnnbb = {'0075': [], '0150': [], '0225': [], '0300': [], '0375': [], '0450': []}
            # Example Set 1 2 3 4
            for i in range(len(new_examples)):
                new_examples_ = torch.stack([new_examples[i]])
                new_labels_ = torch.stack([new_labels[i]])
                new_lengths_ = torch.stack([new_lengths[i]])
                original_labels_ = torch.stack([original_labels[i]])

                new_examples_2 = torch.stack([new_examples2[i]])
                new_labels_2 = torch.stack([new_labels2[i]])
                original_labels_2 = torch.stack([original_labels2[i]])
    
                if use_cuda:
                    new_examples_ = new_examples_.cuda()
                    new_labels_ = new_labels_.cuda()
                    new_lengths_ = new_lengths_.cuda()
                    original_labels_ = original_labels_.cuda()
                    new_examples_2 = new_examples_2.cuda()
                    new_labels_2 = new_labels_2.cuda()
                    original_labels_2 = original_labels_2.cuda()

                data_, lengths_, target_, orig = new_examples_.to(device), new_lengths_.to(device), new_labels_.to(device), original_labels_.to(device)
                data2_, lengths_, target2_, orig2 = new_examples_2.to(device), new_lengths_.to(device), new_labels_2.to(device), original_labels_2.to(device)
                X, y, z = Variable(data_, requires_grad=True), Variable(target_), Variable(orig)
                X2, y2, z2 = Variable(data2_, requires_grad=True), Variable(target_), Variable(orig2)
                X3, y3, z3 = Variable(data_, requires_grad=True), Variable(target2_), Variable(orig)
                X4, y4, z4 = Variable(data2_, requires_grad=True), Variable(target2_), Variable(orig2)

                rnnwbloss, cnnbbloss = _pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.075, i)
                cnnwbloss, rnnbbloss = _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.075, i)
                rnnwbloss2, cnnbbloss2 = _pgd_whitebox(model, X2, lengths, y2, orig2, "rnn", model2, 0.075, i)
                cnnwbloss2, rnnbbloss2 = _pgd_whitebox(model2, X2, lengths, y2, orig2, "cnn", model, 0.075, i)
                rnnwbloss3, cnnbbloss3 = _pgd_whitebox(model, X3, lengths, y3, orig, "rnn", model2, 0.075, i)
                cnnwbloss3, rnnbbloss3 = _pgd_whitebox(model2, X3, lengths, y3, orig, "cnn", model, 0.075, i)
                rnnwbloss4, cnnbbloss4 = _pgd_whitebox(model, X4, lengths, y4, orig2, "rnn", model2, 0.075, i)
                cnnwbloss4, rnnbbloss4 = _pgd_whitebox(model2, X4, lengths, y4, orig2, "cnn", model, 0.075, i)
                #print(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item())
                
                rnnwb['0075'].append(max(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item()))
                cnnwb['0075'].append(max(cnnwbloss.item(), cnnwbloss2.item(), cnnwbloss3.item(), cnnwbloss4.item()))
                rnnbb['0075'].append(max(rnnbbloss.item(), rnnbbloss2.item(), rnnbbloss3.item(), rnnbbloss4.item()))
                cnnbb['0075'].append(max(cnnbbloss.item(), cnnbbloss2.item(), cnnbbloss3.item(), cnnbbloss4.item()))
                #print(rnnwb, cnnwb, rnnbb, cnnbb) 
                rnnwbloss, cnnbbloss = _pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.150, i)
                cnnwbloss, rnnbbloss = _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.150, i)
                rnnwbloss2, cnnbbloss2 = _pgd_whitebox(model, X2, lengths, y2, orig2, "rnn", model2, 0.150, i)
                cnnwbloss2, rnnbbloss2 = _pgd_whitebox(model2, X2, lengths, y2, orig2, "cnn", model, 0.150, i)
                rnnwbloss3, cnnbbloss3 = _pgd_whitebox(model, X3, lengths, y3, orig, "rnn", model2, 0.150, i)
                cnnwbloss3, rnnbbloss3 = _pgd_whitebox(model2, X3, lengths, y3, orig, "cnn", model, 0.150, i)
                rnnwbloss4, cnnbbloss4 = _pgd_whitebox(model, X4, lengths, y4, orig2, "rnn", model2, 0.150, i)
                cnnwbloss4, rnnbbloss4 = _pgd_whitebox(model2, X4, lengths, y4, orig2, "cnn", model, 0.150, i)
                #print(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item())
                rnnwb['0150'].append(max(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item()))
                cnnwb['0150'].append(max(cnnwbloss.item(), cnnwbloss2.item(), cnnwbloss3.item(), cnnwbloss4.item()))
                rnnbb['0150'].append(max(rnnbbloss.item(), rnnbbloss2.item(), rnnbbloss3.item(), rnnbbloss4.item()))
                cnnbb['0150'].append(max(cnnbbloss.item(), cnnbbloss2.item(), cnnbbloss3.item(), cnnbbloss4.item()))
                #print(rnnwb, cnnwb, rnnbb, cnnbb)
                rnnwbloss, cnnbbloss = _pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.225, i)
                cnnwbloss, rnnbbloss = _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.225, i)
                rnnwbloss2, cnnbbloss2 = _pgd_whitebox(model, X2, lengths, y2, orig2, "rnn", model2, 0.225, i)
                cnnwbloss2, rnnbbloss2 = _pgd_whitebox(model2, X2, lengths, y2, orig2, "cnn", model, 0.225, i)
                rnnwbloss3, cnnbbloss3 = _pgd_whitebox(model, X3, lengths, y3, orig, "rnn", model2, 0.225, i)
                cnnwbloss3, rnnbbloss3 = _pgd_whitebox(model2, X3, lengths, y3, orig, "cnn", model, 0.225, i)
                rnnwbloss4, cnnbbloss4 = _pgd_whitebox(model, X4, lengths, y4, orig2, "rnn", model2, 0.225, i)
                cnnwbloss4, rnnbbloss4 = _pgd_whitebox(model2, X4, lengths, y4, orig2, "cnn", model, 0.225, i)
                #print(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item())
                rnnwb['0225'].append(max(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item()))
                cnnwb['0225'].append(max(cnnwbloss.item(), cnnwbloss2.item(), cnnwbloss3.item(), cnnwbloss4.item()))
                rnnbb['0225'].append(max(rnnbbloss.item(), rnnbbloss2.item(), rnnbbloss3.item(), rnnbbloss4.item()))
                cnnbb['0225'].append(max(cnnbbloss.item(), cnnbbloss2.item(), cnnbbloss3.item(), cnnbbloss4.item()))
                #print(rnnwb, cnnwb, rnnbb, cnnbb)
                rnnwbloss, cnnbbloss = _pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.300, i)
                cnnwbloss, rnnbbloss = _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.300, i)
                rnnwbloss2, cnnbbloss2 = _pgd_whitebox(model, X2, lengths, y2, orig2, "rnn", model2, 0.300, i)
                cnnwbloss2, rnnbbloss2 = _pgd_whitebox(model2, X2, lengths, y2, orig2, "cnn", model, 0.300, i)
                rnnwbloss3, cnnbbloss3 = _pgd_whitebox(model, X3, lengths, y3, orig, "rnn", model2, 0.300, i)
                cnnwbloss3, rnnbbloss3 = _pgd_whitebox(model2, X3, lengths, y3, orig, "cnn", model, 0.300, i)
                rnnwbloss4, cnnbbloss4 = _pgd_whitebox(model, X4, lengths, y4, orig2, "rnn", model2, 0.300, i)
                cnnwbloss4, rnnbbloss4 = _pgd_whitebox(model2, X4, lengths, y4, orig2, "cnn", model, 0.300, i)
                #print(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item())
                rnnwb['0300'].append(max(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item()))
                cnnwb['0300'].append(max(cnnwbloss.item(), cnnwbloss2.item(), cnnwbloss3.item(), cnnwbloss4.item()))
                rnnbb['0300'].append(max(rnnbbloss.item(), rnnbbloss2.item(), rnnbbloss3.item(), rnnbbloss4.item()))
                cnnbb['0300'].append(max(cnnbbloss.item(), cnnbbloss2.item(), cnnbbloss3.item(), cnnbbloss4.item()))
                #print(rnnwb, cnnwb, rnnbb, cnnbb)
                rnnwbloss, cnnbbloss = _pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.375, i)
                cnnwbloss, rnnbbloss = _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.375, i)
                rnnwbloss2, cnnbbloss2 = _pgd_whitebox(model, X2, lengths, y2, orig2, "rnn", model2, 0.375, i)
                cnnwbloss2, rnnbbloss2 = _pgd_whitebox(model2, X2, lengths, y2, orig2, "cnn", model, 0.375, i)
                rnnwbloss3, cnnbbloss3 = _pgd_whitebox(model, X3, lengths, y3, orig, "rnn", model2, 0.375, i)
                cnnwbloss3, rnnbbloss3 = _pgd_whitebox(model2, X3, lengths, y3, orig, "cnn", model, 0.375, i)
                rnnwbloss4, cnnbbloss4 = _pgd_whitebox(model, X4, lengths, y4, orig2, "rnn", model2, 0.375, i)
                cnnwbloss4, rnnbbloss4 = _pgd_whitebox(model2, X4, lengths, y4, orig2, "cnn", model, 0.375, i)
                #print(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item())
                rnnwb['0375'].append(max(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item()))
                cnnwb['0375'].append(max(cnnwbloss.item(), cnnwbloss2.item(), cnnwbloss3.item(), cnnwbloss4.item()))
                rnnbb['0375'].append(max(rnnbbloss.item(), rnnbbloss2.item(), rnnbbloss3.item(), rnnbbloss4.item()))
                cnnbb['0375'].append(max(cnnbbloss.item(), cnnbbloss2.item(), cnnbbloss3.item(), cnnbbloss4.item()))
                #print(rnnwb, cnnwb, rnnbb, cnnbb)
                rnnwbloss, cnnbbloss = _pgd_whitebox(model, X, lengths, y, orig, "rnn", model2, 0.450, i)
                cnnwbloss, rnnbbloss = _pgd_whitebox(model2, X, lengths, y, orig, "cnn", model, 0.450, i)
                rnnwbloss2, cnnbbloss2 = _pgd_whitebox(model, X2, lengths, y2, orig2, "rnn", model2, 0.450, i)
                cnnwbloss2, rnnbbloss2 = _pgd_whitebox(model2, X2, lengths, y2, orig2, "cnn", model, 0.450, i)
                rnnwbloss3, cnnbbloss3 = _pgd_whitebox(model, X3, lengths, y3, orig, "rnn", model2, 0.450, i)
                cnnwbloss3, rnnbbloss3 = _pgd_whitebox(model2, X3, lengths, y3, orig, "cnn", model, 0.450, i)
                rnnwbloss4, cnnbbloss4 = _pgd_whitebox(model, X4, lengths, y4, orig2, "rnn", model2, 0.450, i)
                cnnwbloss4, rnnbbloss4 = _pgd_whitebox(model2, X4, lengths, y4, orig2, "cnn", model, 0.450, i)
                #print(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item())
                rnnwb['0450'].append(max(rnnwbloss.item(), rnnwbloss2.item(), rnnwbloss3.item(), rnnwbloss4.item()))
                cnnwb['0450'].append(max(cnnwbloss.item(), cnnwbloss2.item(), cnnwbloss3.item(), cnnwbloss4.item()))
                rnnbb['0450'].append(max(rnnbbloss.item(), rnnbbloss2.item(), rnnbbloss3.item(), rnnbbloss4.item()))
                cnnbb['0450'].append(max(cnnbbloss.item(), cnnbbloss2.item(), cnnbbloss3.item(), cnnbbloss4.item()))
                #print("RNN Whitebox: ", rnnwb)
                #print("RNN Blackbox: ", rnnbb)
                #print("CNN Whitebox: ", cnnwb)
                #print("CNN Blackbox: ", cnnbb)
                #print(rnnwb, cnnwb, rnnbb, cnnbb)
                rnnwbs[w] = rnnwb
                cnnwbs[w] = cnnwb
                rnnbbs[w] = rnnbb
                cnnbbs[w] = cnnbb
            """  
        print("RNN Whitebox: ", rnnwbs)
        print("RNN Blackbox: ", rnnbbs)
        print("CNN Whitebox: ", cnnwbs)
        print("CNN Blackbox: ", cnnbbs)
    # ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
