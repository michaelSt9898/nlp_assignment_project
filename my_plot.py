from matplotlib import pyplot as plt
from pandas import read_csv


def plot_log_results(filename):
    log_frame = read_csv('log/'+filename, sep=';')
    plt.figure(figsize=[9, 4])

    plt.subplot(221)
    plt.title("Train Accuracy")
    plt.grid()
    plt.xlim((0, log_frame['epoch'].iat[-1]))
    plt.plot(log_frame['epoch'], log_frame['accuracy'], "b")

    plt.subplot(222)
    plt.title("Train Loss")
    plt.grid()
    plt.xlim((0, log_frame['epoch'].iat[-1]))
    plt.plot(log_frame['epoch'], log_frame['loss'], "r")

    # validation
    plt.subplot(223)
    plt.title("Validation Accuracy")
    plt.grid()
    plt.xlim((0, log_frame['val_accuracy'].iat[-1]))
    plt.plot(log_frame['epoch'], log_frame['val_accuracy'], "b")

    plt.subplot(224)
    plt.title("Validation loss")
    plt.grid()
    plt.xlim((0, log_frame['val_loss'].iat[-1]))
    plt.plot(log_frame['epoch'], log_frame['val_loss'], "r")
