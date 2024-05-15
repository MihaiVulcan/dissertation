import random
from credit_app import credit

max_posible_credit = 2430000
min_posible_credit = 3500

def write_to_file(dataset, file):
    with open(file, "w") as f:
        f.write("age,yearly_salary,year_wanted,preffed_custommer,max_credit,cat\n")
        for data in dataset:
            f.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "\n")

def main(no_cases, no_categories, file):

    #generate categories:
    spred = int((max_posible_credit-min_posible_credit)/no_categories)
    categories = [0]
    for i in range(no_categories):
        categories.append(categories[i]+spred)

    print(categories)

    dataset = []
    no_each_cat = [0] * (no_categories+1)
    while(sum(no_each_cat)<no_cases):
        age = random.randrange(20,65)
        yearly_salary = random.randrange(10000,100000)
        years_wanted = random.randrange(1,30)
        preffered_customer = random.randint(0,1)
        res_credit = credit(age, yearly_salary, years_wanted, preffered_customer)
        #determine category:
        cat = 0
        while res_credit > categories[cat]:
            cat += 1

        if(no_each_cat[cat]<no_cases/no_categories):
            no_each_cat[cat] += 1
            dataset.append([age, yearly_salary, years_wanted, preffered_customer, res_credit, cat])

    print(no_each_cat)

    write_to_file(dataset, file)

main(800, 4, "dataset_generator/credit_dataset/data/dataset.csv")