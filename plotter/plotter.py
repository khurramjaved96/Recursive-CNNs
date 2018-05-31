''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')

MEDIUM_SIZE = 18

font = {'family': 'sans-serif',
        'weight': 'bold'}

matplotlib.rc('xtick', labelsize=MEDIUM_SIZE)
matplotlib.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels

# matplotlib.rc('font', **font)
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})


class Plotter():
    def __init__(self):
        import itertools
        # plt.figure(figsize=(12, 9))
        self.marker = itertools.cycle(('o', '+', "v", "^", "8", '.', '*'))
        self.handles = []
        self.lines = itertools.cycle(('--', '-.', '-', ':'))

    def plot(self, x, y, xLabel="Number of Classes", yLabel="Accuracy %", legend="none", title=None, error=None):
        self.x = x
        self.y = y
        plt.grid(color='0.89', linestyle='--', linewidth=1.0)
        if error is None:
            l, = plt.plot(x, y, linestyle=next(self.lines), marker=next(self.marker), label=legend, linewidth=3.0)
        else:
            l = plt.errorbar(x, y, yerr=error, capsize=4.0, capthick=2.0, linestyle=next(self.lines),
                             marker=next(self.marker), label=legend, linewidth=3.0)

        self.handles.append(l)
        self.x_label = xLabel
        self.y_label = yLabel
        if title is not None:
            plt.title(title)

    def save_fig(self, path, xticks=105, title=None, yStart=0, xRange=0, yRange=10):
        if title is not None:
            plt.title(title)
        plt.legend(handles=self.handles)
        plt.ylim((yStart, 100 + 0.2))
        plt.xlim((0, xticks + .2))
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.yticks(list(range(yStart, 101, yRange)))
        print(list(range(yStart, 105, yRange)))
        plt.xticks(list(range(0, xticks + 1, xRange + int(xticks / 10))))
        plt.savefig(path + ".eps", format='eps')
        plt.gcf().clear()

    def save_fig2(self, path, xticks=105):
        plt.legend(handles=self.handles)
        plt.xlabel("Memory Budget")
        plt.ylabel("Average Incremental Accuracy")
        plt.savefig(path + ".jpg")
        plt.gcf().clear()

    def plotMatrix(self, epoch, path, img):

        plt.imshow(img, cmap='plasma', interpolation='nearest')
        plt.colorbar()
        plt.savefig(path + str(epoch) + ".svg", format='svg')
        plt.gcf().clear()

    def saveImage(self, img, path, epoch):
        from PIL import Image
        im = Image.fromarray(img)
        im.save(path + str(epoch) + ".jpg")


if __name__ == "__main__":
    pl = Plotter()
    pl.plot([1, 2, 3, 4], [2, 3, 6, 2])
    pl.save_fig("test.jpg")
