import os
if __name__ == '__main__':
    file_name = "magazine001"
    dir = "../testDataset/"
    import csv

    with open('gt.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for folder in os.listdir(dir):
            dir_temp = dir+folder+"/"
            for file in os.listdir(dir_temp):
                print file
                from subprocess import call
                if(file.endswith(".avi")):
                    call("mkdir "+folder, shell=True)
                    call("cd "+folder+" && mkdir "+file, shell=True)
                    call("ls", shell=True)

                    location=  dir+folder+"/"+file
                    gt_address =  "cp " + location[0:-4]+".gt.xml "+folder+"/"+file+"/"+file+".gt"
                    call(gt_address ,shell = True)
                    command = "ffmpeg -i "+location+ " "+folder+"/"+file+"/%3d.jpg"
                    print command
                    call(command, shell=True)


