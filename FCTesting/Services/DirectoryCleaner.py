import shutil
import os


def clean_dir():
    shutil.rmtree('/home/linuxsagar/tempTest/')
    os.mkdir('/home/linuxsagar/tempTest/')


def clean_graph_dir():
    # shutil.rmtree('/home/linuxsagar/PycharmProjects/FruitClassifierTesting/graphs/')
    # os.mkdir('')
    # os.remove()
    os.system("rm -rf /home/linuxsagar/PycharmProjects/FruitClassifierTesting/FCTesting/static/histo.png")