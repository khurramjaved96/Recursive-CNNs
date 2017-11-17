import trainer.trainerFactory as tF
def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainData", help="npy file for training data")
    parser.add_argument("-v", "--validateData", help="npy file for validation set")
    parser.add_argument("-c", "--checkpointDir", help="Directory to store checkpoints")
    return  parser.parse_args()


if __name__=="__main__":
    args = argsProcessor()
    model = tF.trainerFactory.getTrainer("documentDetector", args.trainData, args.validateData, args.checkpointDir)
    model.loadData()
    model.setupModel()
    model.train(1000)
