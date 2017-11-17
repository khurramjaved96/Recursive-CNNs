import trainer.trainer as trainer

def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataPath", help="Path containing correctly formated data")
    parser.add_argument("-t", "--type", help="decide if corner or four corner model should be trained")
    return  parser.parse_args()


if __name__=="__main__":
    args = argsProcessor()