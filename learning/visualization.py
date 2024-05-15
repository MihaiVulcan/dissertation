import matplotlib.pyplot as plt

def flatten(xss):
    return [x for xs in xss for x in xs]

def plot_actual_predicted(actual, predicted):
    actual = flatten(actual)
    predicted = flatten(predicted)
    actual, predicted = zip(*sorted(zip(actual, predicted)))
    plt.plot(flatten(actual))
    plt.plot(flatten(predicted))
    plt.show()