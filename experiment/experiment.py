''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import json
import os
import subprocess


class experiment:
    '''
    Class to store results of any experiment 
    '''

    def __init__(self, name, args, output_dir="../"):
        self.gitHash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
        print(self.gitHash)
        if not args is None:
            self.name = name
            self.params = vars(args)
            self.results = {}
            self.dir = output_dir

            import datetime
            now = datetime.datetime.now()
            rootFolder = str(now.day) + str(now.month) + str(now.year)
            if not os.path.exists(output_dir + rootFolder):
                os.makedirs(output_dir + rootFolder)
            self.name = rootFolder + "/" + self.name
            ver = 0

            while os.path.exists(output_dir + self.name + "_" + str(ver)):
                ver += 1

            os.makedirs(output_dir + self.name + "_" + str(ver))
            self.path = output_dir + self.name + "_" + str(ver) + "/" + name

            self.results["Temp Results"] = [[1, 2, 3, 4], [5, 6, 2, 6]]

    def store_json(self):
        with open(self.path + "JSONDump.txt", 'w') as outfile:
            json.dump(json.dumps(self.__dict__), outfile)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iCarl2.0')
    args = parser.parse_args()
    e = experiment("TestExperiment", args)
    e.store_json()
