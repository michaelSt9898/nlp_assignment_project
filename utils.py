import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_prediction(y, y_pred, classes):
    punct_marcs = ["''", ",", ".", ":", '``', '0']
    punct_indexes = [i for i, c in enumerate(classes) if c in punct_marcs]

    assert y.shape == y_pred.shape

    y = y.reshape(y.shape[0]*y.shape[1], -1)
    y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1], -1)

    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)

    total = 0
    correct = 0
    for i, el in enumerate(y):
        if el in punct_indexes:
            continue
        total += 1
        if el == y_pred[i]:
            correct += 1

    # correct = np.count_nonzero(y==y_pred)
    print("Accuracy: {p:.2f} %".format(p=correct/total*100))

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict(x)  # batch?
    if not binary:
        y_pred = np.argmax(y_pred, axis=1)

    print(y_pred.shape)

    # 3. Print accuracy score
    print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

    print("")

    # list english punctuation symbols
    eng_punc = ['.', ',', '"', "'","''",  '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '%', '@', '#', '$', '^', '&', '*', '-', '_', '=', '+', '<', '>', '/', '~', '`', '|', '\\', ' ']
    # create list of labels that are not punctuation
    no_punc_labels = np.where(~np.isin(classes, eng_punc))[0]


    # 4. Print classification report
    print("")
    print("Classification Report without punctuation")
    print(classification_report(y_true, y_pred, labels=no_punc_labels, target_names=classes[no_punc_labels], digits=5, zero_division=0))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)
    # plot_confusion_matrix(cnf_matrix,classes=classes)