from utils import parser
from utils import utils
from model.valstm import VALSTM
from model.vacnn import VACNN
import torch
import torch.nn as nn
import os
import sys
import data_loader
import numpy as np
import random
import time
from tqdm import tqdm
def train(model, train_loader, criterion, optimizer, epoch, params):
    model.train()
    total_train_loss = 0.
    correct = 0
    with tqdm(total=len(train_loader)) as t:
        for batch_id, (data, target,frame_num) in enumerate(train_loader):
            # print("frame_num.size(): " + str(frame_num.size()))
            data, target, frame_num = data.float().cuda(), target.long().cuda(), frame_num.long().cuda() # important float() and long() operation
            output = model(data, target, frame_num)
            loss = criterion(output, target)
            total_train_loss += loss
            pred = output.argmax(dim = 1, keepdim = True) # get the index of max log-prob
            correct += pred.eq(target.view_as(pred)).sum().item()
            optimizer.zero_grad()  
            loss.backward()
            if params.use_gradclip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradnorm)
            optimizer.step() 
            t.update()

    train_loss = total_train_loss / len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    print('Training: Epoch:{:>3}, Total loss: {:.4f}, Training Accuracy: {}/{} ({:.1f}%)'.format(epoch,
    total_train_loss, correct, len(train_loader.dataset), acc))
    return acc, total_train_loss
def test(model, test_loader, criterion, epoch, best = False):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target, frame_num in test_loader:
            data, target, frame_num = data.float().cuda(), target.long().cuda(), frame_num.long().cuda()
            output = model(data, target, frame_num)
            test_loss += criterion(output, target)
            pred = output.argmax(dim = 1, keepdim = True) # get the index of max log-prob
            correct += pred.eq(target.view_as(pred)).sum().item()

    
    # test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    if not best:   
        print('Epoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        # print("Epoch:%3d  Average loss:%.2f  Accuracy:%.2f" % (epoch, test_loss, test_acc), file = logfile)
        with open(logpath, 'a+') as logfile:
            print('Epoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(epoch,
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), file = logfile)
    else:
        print("BestPerformance:\nEpoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)".format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        with open(logpath, 'a+') as logfile:
            print("BestPerformance:\nEpoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)".format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), file = logfile)
    return test_acc, test_loss

def visualize(model, test_loader):
    model.eval()
    print("==> print model properties")
    print(model.__dict__)
    with torch.no_grad():
        for data, target, frame_num in test_loader:
            data, target, frame_num = data.float().cuda(), target.long().cuda(), frame_num.long().cuda()
            data_transform = model.va_subnet(data) # (batch, seq_len, 150)
            print("data[0][0]: " +str(data[0][0]))
            print("data_transform[0][0]: " + str(data_transform[0][0]))
            print("dif[0][0]" + str(data_transform[0][0] - data[0][0]))
            # draw(data[0].data.cpu().numpy(), data_transform[0].data.cpu().numpy(), 
            # target[0].data.cpu().numpy(),frame_num[0].data.cpu().numpy())
            break

def draw(data, data_transform, target, frame_num):
    """
        Arguments:s
        data: (seq_len, 150)
        data_transform: (seq_len, 150)
        target: (1,)
        frame_num(1,)
    """
    target = np.reshape(target, (-1))[0]
    frame_num = np.reshape(frame_num, (-1))[0]
    diff = data - data_transform
    print("data[0,:10,:10]:")
    for i in range(10):
        for j in range(10):
            print(data[i,j], end = ' ')
        print()
    print("data_transform[0,:10,:10]:")
    for i in range(10):
        for j in range(10):
            print(data_transform[i,j], end = ' ')
        print()
    print("diff[0,:10,:10]:")
    for i in range(10):
        for j in range(10):
            print(diff[i,j], end = ' ')
        print()

def save_checkpoint(state):
    torch.save(state,modelpath)

def train_and_evaluate(model, train_loader, test_loader, epochs, optimizer, criterion, params, start_epoch = 1):
    print("==> initiating model")
    model.init_weights()
    print("==> print model properties")
    print(model.__dict__)
    total_loss = 0.
    best_acc = 0.
    best_epoch = 1
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    if args.lr_decay:
        if args.optim_mode=='max':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', 
            factor=params.decay_rate, patience=params.patience, cooldown = 3, verbose= True)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', 
            factor=params.decay_rate, patience=params.patience, cooldown = 3, verbose= True)
    print("==> training model")
    stages = [int(_) for _ in args.stages]
    epochs_per_stage = [int(_) for _ in args.epochs_per_stage]
    print("stages: " + str(stages))
    # model.switch_stage(stages[0])
    # test_acc, test_loss = test(model, test_loader, criterion, False)
    try:
        epoch = 0
        for i in range(len(stages)):
            stage = stages[i]
            epochs = epochs_per_stage[i]
            model.switch_stage(stage)
            for local_epoch in range(epochs):
                epoch += 1
                start = time.time()
                train_acc, train_loss = train(model, train_loader, criterion, optimizer, epoch, params)
                test_acc, test_loss = test(model, test_loader, criterion, epoch, False)
                if args.lr_decay and i==len(stages)-1:
                    if args.optim_mode=='max':
                        scheduler.step(test_acc)
                    else:
                        scheduler.step(test_loss)
                elapsed_time = time.time() - start
                s = time.strftime("%M:%S", time.gmtime(elapsed_time))
                print("Time for Epoch {:>3}:    {}".format(epoch, s)) 
                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)
                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)
                if test_acc > best_acc:
                    best_epoch = epoch
                    best_acc = test_acc
                    save_checkpoint({
                    'epoch': epoch,
                    'num_classes': 1 + max(train_loader.dataset.label),
                    'hidden': args.hidden_unit,
                    'trans': args.trans,
                    'rota': args.rota,
                    'dropout': args.dropout,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()})
    finally:
        # draw and save loss and acc
        loss_file_name = 'loss_' + args.version 
        acc_file_name = 'acc_' + args.version 
        loss_file_path = os.path.join(args.checkpoint_folder, loss_file_name)
        acc_file_path = os.path.join(args.checkpoint_folder, acc_file_name)
        np.save(loss_file_path + '_train.npy', np.asarray(train_loss_list))
        np.save(loss_file_path + '_test.npy', np.asarray(test_loss_list))
        np.save(acc_file_path + '_train.npy', np.asarray(train_acc_list))
        np.save(acc_file_path + '_test.npy', np.asarray(test_acc_list))

        # best mode output
        print("==> testing model")
        model_info = torch.load(modelpath)
        model.load_state_dict(model_info['state_dict'])
        test_acc, test_loss = test(model, test_loader, criterion, best_epoch, True)
def sync_params_and_args():
    params.dataset_dir = args.dataset_dir
    params.dataset_name = args.dataset_name
    params.use_gradclip = args.gradclip
    params.epochs = args.epochs

if __name__ == '__main__':
    # parse args
    global args
    args = parser.parser.parse_args()
    print("args.downsampling: " + str(args.downsampling))
    print("args.mask_person: " + str(args.mask_person))
    print("args.mask_frame: " + str(args.mask_frame))
    # set logfile, modelpath, jsonfile
    global modelpath
    global logpath
    if not os.path.exists(args.checkpoint_folder):
        os.mkdir(args.checkpoint_folder)
    model_name = 'model' + args.version + '.pt'
    modelpath = os.path.join(args.checkpoint_folder, model_name)
    logpath = os.path.join(args.checkpoint_folder, args.version + ".txt")
    if not args.loadfrompath:
        logfile = open(logpath, 'w+')
        print(sys.argv, file = logfile)
        logfile.close()

    # set jsonfile
    global params
    if 'NTU' in args.dataset_name:
        json_file = os.path.join(args.checkpoint_folder, 'ntu_params.json')
    elif 'PKU' in args.dataset_name:
        json_file = os.path.join(args.checkpoint_folder, 'pkummd_params.json')
    else:
        raise ValueError()
    params = utils.Params(json_file)

    # sync params and args
    sync_params_and_args()

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # set gpu
    print("gpu_id: " + str(args.gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)

    # load dataset
    print("==> loading data")

    train_loader = data_loader.fetch_dataloader('train', params, args)
    test_loader = data_loader.fetch_dataloader('test', params, args)

    input_size = train_loader.dataset.data.shape[-1]
    num_classes = 1 + max(train_loader.dataset.label)

    print("train data shape: " + str(train_loader.dataset.data.shape))
    print("test data shape: " + str(test_loader.dataset.data.shape))

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # load or create model
    if args.backbone=='lstm':
        if args.loadfrompath:
            print("==> loading existing lstm model")
            model_info = torch.load(modelpath)
            model = VALSTM(
                input_size = input_size,
                num_classes = model_info['num_classes'],
                hidden = model_info['hidden'],
                trans = model_info['trans'],
                rota = model_info['rota'],
                dropout = model_info['dropout'],
                pengfei= args.pengfei,
                mean_after_fc = args.mean_after_fc,
                dataset_name= args.dataset_name,
                mask_person = args.mask_person,
                mask_frame = args.mask_frame,
                more_hidden = args.more_hidden
            )
            model.cuda()
            model.load_state_dict(model_info['state_dict'])
            best_acc = model_info['best_acc']
            optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
            optimizer.load_state_dict(model_info['optimizer'])
        else:
            print("==> creating lstm model")
            model = VALSTM(input_size = input_size, 
                num_classes = num_classes, 
                hidden = args.hidden_unit, 
                trans = args.trans, 
                rota = args.rota, 
                dropout = args.dropout,
                pengfei = args.pengfei,
                mean_after_fc = args.mean_after_fc,
                dataset_name= args.dataset_name,
                mask_person = args.mask_person,
                mask_frame = args.mask_frame,
                more_hidden = args.more_hidden,
            )
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    elif args.backbone=='cnn':
        model = VACNN(input_size = input_size, 
                num_classes = num_classes, 
                hidden = args.hidden_unit, 
                trans = args.trans, 
                rota = args.rota, 
                dropout = args.dropout,
                mean_after_fc = args.mean_after_fc,
                dataset_name= args.dataset_name)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)

    # train and test
    if not args.loadfrompath:
        train_and_evaluate(model, train_loader, test_loader, args.epochs, optimizer, criterion, params)
    else:
        print("==> visualizing model")
        visualize(model, test_loader)
        # start_epoch = model_info['epoch']
        # train_and_evaluate(model, train_loader, test_loader, args.epochs, optimizer, criterion, params, start_epoch)



