from matplotlib import pyplot as plt
from pandas import read_csv


def plot_log_results(filename):
    log_frame = read_csv('log/'+filename, sep=';')
    plt.figure(figsize=[9, 4])

    plt.subplot(121)
    plt.grid()
    plt.xlim((0, log_frame['epoch'].iat[-1]))
    plt.title("Accuracy")
    plt.plot(log_frame['epoch'], log_frame['accuracy'], "b")

    plt.subplot(122)
    plt.title("Cross Entropy Loss")
    plt.grid()
    plt.xlim((0, log_frame['epoch'].iat[-1]))
    plt.plot(log_frame['epoch'], log_frame['loss'], "r")
