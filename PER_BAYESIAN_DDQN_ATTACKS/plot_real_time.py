import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import os
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', dest='file', help='file that the data will be extracted',
                        default=None, type=str)
    parser.add_argument('-p', '--path', dest='path', help='path to the file',
                        default='./', type=str)
    args = parser.parse_args()
    return args

def animate(i):
    pullData = open(path,"r").read()
    data = pullData.split('\n')
    xar = []
    yar = []
    for line in data:
        if len(line) > 1:
            y = float(line)
            yar.append(y)

    ax1.clear()
    ax1.plot(yar)


def main():
    file_name = args.file
    global path 
    path = args.path if args.path != './' else args.path + file_name
    path = file_name

    print(path)

    fig = plt.figure()
    global ax1
    ax1 = fig.add_subplot(1,1,1)


    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main()