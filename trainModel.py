''' Incremental-Classifier Learning
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''
from __future__ import print_function

import argparse
import logging
import sys

import torch
import torch.utils.data as td

import dataHandler
import experiment as ex
import model
import plotter as plt
import trainer

import utils.Colorer

logger = logging.getLogger('iCARL')

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.000001, metavar='LR',
                    help='learning rate (default: 2.0)')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30, 40],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--debug', action='store_true', default=True,
                    help='Debug messages')
parser.add_argument('--seed', type=int, default=2323,
                    help='Seeds values to be used')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="../",
                    help='Directory to store the results; a new folder "DDMMYYYY" will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay (L2 penalty).')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="SmartDoc", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument("-i", "--data-dirs", nargs='+', default="/Users/khurramjaved96/documentTest64", help="input Directory of train data")
parser.add_argument("-v", "--validation-dirs", nargs='+', default="/Users/khurramjaved96/documentTest64", help="input Directory of val data")

args = parser.parse_args()

# Define an experiment.
my_experiment = ex.experiment(args.name, args)

# Adding support for logging. A .log is generated with all the logs. Logs are also stored in a temp file one directory
# before the code repository
logger = logging.getLogger('iCARL')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(my_experiment.path + ".log")
fh.setLevel(logging.DEBUG)

fh2 = logging.FileHandler("../temp.log")
fh2.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh2.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(fh2)
logger.addHandler(ch)


args.cuda = not args.no_cuda and torch.cuda.is_available()

dataset = dataHandler.DatasetFactory.get_dataset(args.data_dirs)

dataset_val = dataHandler.DatasetFactory.get_dataset(args.validation_dirs)

# Fix the seed.
seed = args.seed
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

# Loader used for training data
train_dataset_loader = dataHandler.myLoader(dataset.myData, transform=dataset.train_transform,
                                            cuda=args.cuda)

# Loader used for training data
val_dataset_loader = dataHandler.myLoader(dataset_val.myData, transform=dataset.test_transform,
                                            cuda=args.cuda)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Iterator to iterate over training data.
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                         batch_size=args.batch_size, shuffle=True, **kwargs)

# Iterator to iterate over training data.
val_iterator = torch.utils.data.DataLoader(val_dataset_loader,
                                         batch_size=args.batch_size, shuffle=True, **kwargs)

# Get the required model
myModel = model.ModelFactory.get_model(args.model_type)
if args.cuda:
    myModel.cuda()


# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                        weight_decay=args.decay, nesterov=True)

# Trainer object used for training
my_trainer = trainer.Trainer(train_iterator, myModel, args.cuda, optimizer)

# Evaluator
my_eval = trainer.EvaluatorFactory.get_evaluator("mse", args.cuda)
# Running epochs_class epochs
for epoch in range(0, args.epochs):
    logger.info("Epoch : %d", epoch)
    my_trainer.update_lr(epoch, args.schedule, args.gammas)
    my_trainer.train(epoch)
    my_eval.evaluate(my_trainer.model, val_iterator)

torch.save(myModel.state_dict(), my_experiment.path+"ModelState_final")
my_experiment.store_json()

