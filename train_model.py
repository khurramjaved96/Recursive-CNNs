''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

from __future__ import print_function

import argparse

import torch
import torch.utils.data as td
import wandb
from torchvision import transforms

import dataprocessor
import experiment as ex
import model
import trainer
import utils

experiment_names = "Experiment-7-doc"

output_dir = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\experiments"
no_cuda = False
data_dirs = [
    "/home/ubuntu/document_localization/Recursive-CNNs/datasets/augmentations",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/smart-doc-train",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/self_collected",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/kosmos"
]
dataset_type = "document"
validation_dirs = [
    "/home/ubuntu/document_localization/Recursive-CNNs/datasets/augmentations",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/smart-doc-train",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/self_collected",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/kosmos"
]
loader = "ram"

model_type = "resnet"

pretrain = False

lr = 0.005
batch_size = 500
seed = 42

decay = 0.00001

epochs = 75
cuda = not no_cuda and torch.cuda.is_available()

arguments = {
    "experiment_names": experiment_names,
    "output_dir": output_dir,
    "cuda": cuda,
    "data_dirs": data_dirs,
    "dataset_type": dataset_type,
    "validation_dirs": validation_dirs,
    "loader": loader,
    "model_type": model_type,
    "pretrain": pretrain,
    "lr": lr,
    "batch_size": batch_size,
    "seed": seed,
    "decay": decay,
    "epochs": epochs,
}
# wandb.login(key=[your_api_key])
wandb.init(project="document-detection",
           entity="kosmos-randd",
           config=arguments)

wandb.run.name = wandb.run.name + "-" + experiment_names
# Define an experiment.
my_experiment = ex.experiment(experiment_names, arguments, output_dir)

# Add logging support
logger = utils.utils.setup_logger(my_experiment.path)

#%%
dataset = dataprocessor.DatasetFactory.get_dataset(data_dirs, dataset_type, "train.csv")
#%%
dataset_val = dataprocessor.DatasetFactory.get_dataset(validation_dirs, dataset_type, "test.csv")
#%%


transforms.Compose([transforms.Resize([32, 32]),
                    transforms.ColorJitter(.5, .5, .5, 0.2),
                    transforms.ToTensor()])


train_dataset_loader = dataprocessor.LoaderFactory.get_loader(loader, dataset.myData,
                                                              transform=dataset.train_transform,
                                                              cuda=cuda)
# Loader used for training data
val_dataset_loader = dataprocessor.LoaderFactory.get_loader(loader, dataset_val.myData,
                                                            transform=dataset.test_transform,
                                                            cuda=cuda)
kwargs = {'num_workers': 30, 'pin_memory': True} if cuda else {}

# Iterator to iterate over training data.
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=batch_size, shuffle=True, **kwargs)
# Iterator to iterate over training data.
val_iterator = torch.utils.data.DataLoader(val_dataset_loader,
                                           batch_size=batch_size, shuffle=True, **kwargs)

# Get the required model
myModel = model.ModelFactory.get_model(model_type, dataset_type)

myModel.load_state_dict(torch.load(r"/home/ubuntu/document_localization/Recursive-CNNs/experiments3082024/Experiment-3-doc_0/Experiment-3-docdocument_resnet.pb", map_location='cpu'))

if cuda:
    myModel.cuda()

# Should I pretrain the model on CIFAR?
if pretrain:
    trainset = dataprocessor.DatasetFactory.get_dataset(None, "CIFAR")
    train_iterator_cifar = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # Define the optimizer used in the experiment
    cifar_optimizer = torch.optim.Adam(myModel.parameters(), lr=lr, weight_decay=decay)

    # Trainer object used for training
    cifar_trainer = trainer.CIFARTrainer(train_iterator_cifar, myModel, cuda, cifar_optimizer)

    for epoch in range(0, 70):
        logger.info("Epoch : %d", epoch)
        cifar_trainer.train(epoch)

    # Freeze the model
    counter = 0
    for experiment_names, param in myModel.named_parameters():
        # Getting the length of total layers so I can freeze x% of layers
        gen_len = sum(1 for _ in myModel.parameters())
        if counter < int(gen_len * 0.5):
            param.requires_grad = False
            logger.warning(experiment_names)
        else:
            logger.info(experiment_names)
        counter += 1

# Define the optimizer used in the experiment
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, myModel.parameters()), lr=lr, weight_decay=decay)

# Trainer object used for training
my_trainer = trainer.Trainer_with_class(train_iterator, myModel, cuda, optimizer)

# Evaluator
my_eval = trainer.EvaluatorFactory.get_evaluator("rmse", cuda)

max_accuracy=0

for epoch in range(0, epochs):
    logger.info("Epoch : %d", epoch)
    my_trainer.train(epoch)
    accuracy,accuracy_4,accuracy_3=my_eval.evaluate(my_trainer.model, val_iterator, epoch, "val_", False)

    if accuracy>max_accuracy:
        max_accuracy=accuracy
        torch.save(myModel.state_dict(), my_experiment.path + dataset_type + "_" + model_type+f"best_{epoch}" + ".pb")
        wandb.log({
            "Max_accuracy":max_accuracy,
            "Max_accuracy_4_corners":accuracy_4,
            "Max_accuracy_3_corners":accuracy_3,
            "epoch":epoch
                   })
# Final evaluation on test set
my_eval.evaluate(my_trainer.model, val_iterator, epoch, "test_", True)

torch.save(myModel.state_dict(), my_experiment.path + dataset_type + "_" + model_type + ".pb")
my_experiment.store_json()
wandb.finish()
