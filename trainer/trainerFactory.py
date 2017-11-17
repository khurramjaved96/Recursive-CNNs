import trainer
class trainerFactory:
    @staticmethod
    def getTrainer(type,trainDir, validateDir, checkpointDir):
        if type=="documentDetector":
            return trainer.documentDetector(trainDir, validateDir, checkpointDir)
        else:
            return trainer.cornerDetector