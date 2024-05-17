from ann import train_ann
from rbf import train_rbf
from multiprocessing import freeze_support


def l4 (type, function, list_of_epochs, list_of_hidden_layer_sizes):
    result = []
    for epochs in list_of_epochs:
        for hidden_layer_size in list_of_hidden_layer_sizes:
            res = function(type , epochs, hidden_layer_size, 0.001, 1)
            result.append([epochs, hidden_layer_size, res])
    return result

def main(type, function, list_of_epochs, list_of_hidden_layer_sizes):
    if __name__ == '__main__':
        freeze_support()
        return l4(type, function, list_of_epochs,list_of_hidden_layer_sizes)

# res_ann_triangle = main("triangle", train_ann, [50,100], [25, 50])


# res_ann_credit = main("credit", train_ann, [50,100], [25, 50])


# res_ann_heart_risk= main("heart_risk", train_ann, [50,100], [25, 50])

res_rbf_triangle = main("triangle", train_ann, [50,100], [200, 350])


res_rbf_credit = main("credit", train_ann, [50,100], [200, 350])


res_rbf_heart_risk = main("heart_risk", train_ann, [50,100], [200, 350])

# print(res_ann_triangle)
# print(res_ann_credit)
# print(res_ann_heart_risk)
print(res_rbf_triangle)
print(res_rbf_credit)
print(res_rbf_heart_risk)