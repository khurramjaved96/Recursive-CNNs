import trainer
class trainerFactory:
    @staticmethod
    def getTrainer(type,trainDir, validateDir, checkpointDir, concatinateFeatures):
        if type=="documentDetector":
            return trainer.documentDetector(trainDir, validateDir, checkpointDir, concatinateFeatures = concatinateFeatures)
        elif type=="cornerDetector":
            return trainer.cornerDetector