import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot


def clean_dir():
    shutil.rmtree('/home/linuxsagar/tempTest/')
    os.mkdir('/home/linuxsagar/tempTest/')


def clean_graph_dir():
    os.system("rm -rf /home/linuxsagar/PycharmProjects/FruitClassifierTesting/FCTesting/static/histo.png")


def make_histogram(i_hist, i_hufe, i_ehfe, i_hhfe, dataDir):
    fig, axes = plt.subplots(nrows=4, ncols=4)
    fig.tight_layout()

    plt.subplot(2, 2, 1)
    plt.title("Input Image Histogram")
    plot(i_hist)
    plt.xlabel("Frequency")
    plt.ylabel("Intensities")

    plt.subplot(2, 2, 2)
    plt.title("Hue Histogram")
    plt.xlabel("Hue level")
    plt.ylabel("Values")
    plot(i_hufe)

    plt.subplot(2, 2, 3)
    plt.title("Edge Histogram")
    plt.xlabel("Edge vectors")
    plt.ylabel("Values")
    plot(i_ehfe)

    plt.subplot(2, 2, 4)
    plt.title("Haar-Like feature Histogram")
    plt.xlabel("Haar-Like features")
    plt.ylabel("Values")
    plot(i_hhfe)

    plt.savefig(os.path.join(dataDir, "static/histo.png"), dpi=100)