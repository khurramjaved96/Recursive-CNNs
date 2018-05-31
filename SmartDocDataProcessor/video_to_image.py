import os
import tqdm.tqdm as tqdm

def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataPath", help="path to main data folder")
    parser.add_argument("-o", "--outputPath", help="output data")
    return  parser.parse_args()


if __name__ == '__main__':
    args = argsProcessor()
    dir = args.dataPath
    output=args.outputPath
    if (not os.path.isdir(output)):
        os.mkdir(output)
    import csv


    for folder in os.listdir(dir):
        if os.path.isdir(dir+"/"+folder):
            dir_temp = dir+folder+"/"
            for file in os.listdir(dir_temp):
                print (file)
                from subprocess import call
                if(file.endswith(".avi")):
                    call("mkdir "+output + folder, shell=True)
                    if(os.path.isdir(output+folder+"/"+file)):
                        print ("Folder already exist")
                    else:
                        call("cd "+output+folder+" && mkdir "+file, shell=True)
                        call("ls", shell=True)

                        location=  dir+folder+"/"+file
                        gt_address =  "cp " + location[0:-4]+".gt.xml "+output+ folder+"/"+file+"/"+file+".gt"
                        call(gt_address ,shell = True)
                        command = "ffmpeg -i "+location+ " "+output+folder+"/"+file+"/%3d.jpg"
                        print (command)
                        call(command, shell=True)


