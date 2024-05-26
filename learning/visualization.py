import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def flatten(xss):
    return [x for xs in xss for x in xs]

def plot_actual_predicted(actual, predicted):
    actual = flatten(actual)
    predicted = flatten(predicted)
    actual, predicted = zip(*sorted(zip(actual, predicted)))
    plt.figure(1)
    plt.plot(flatten(actual))
    plt.plot(flatten(predicted))
    plt.show()


def plot_roc_curve(all_labels, all_outputs):
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure(2)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()