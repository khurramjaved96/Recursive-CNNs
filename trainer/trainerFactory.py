import trainer.trainer as trainer
class trainerFactory:
    @staticmethod
    def getTrainer(type="corner"):
        if type=="corner":
            return trainer.documentDetector
        else:
            return trainer.cornerDetector