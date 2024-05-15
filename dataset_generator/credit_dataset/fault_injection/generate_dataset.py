import random
from credit_app_no_fault import credit_no_fault
from credit_app_faulty_1 import credit_fault_1
from credit_app_faulty_2 import credit_fault_2
from credit_app_faulty_3 import credit_fault_3
from credit_app_faulty_4 import credit_fault_4

max_posible_credit = 2430000
min_posible_credit = 3500

def write_to_file(dataset, file):
    with open(file, "w") as f:
        f.write("age,yearly_salary,year_wanted,preffed_custommer,max_credit,cat,actual_cat\n")
        for data in dataset:
            f.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + "\n")

def main(no_cases, no_categories, faulty_functions, file):

    #generate categories:
    spred = int((max_posible_credit-min_posible_credit)/no_categories)
    categories = [0]
    for i in range(no_categories):
        categories.append(categories[i]+spred)

    print(categories)

    dataset = []
    dataset_faults = []
    no_each_cat = [0] * (no_categories+1)
    while(sum(no_each_cat)<no_cases):
        age = random.randrange(20,65)
        yearly_salary = random.randrange(10000,100000)
        years_wanted = random.randrange(1,30)
        preffered_customer = random.randint(0,1)
        res_credit = credit_no_fault(age, yearly_salary, years_wanted, preffered_customer)
        #determine category:
        cat = 0
        while res_credit > categories[cat]:
            cat += 1

        if(no_each_cat[cat]<no_cases/no_categories):
            no_each_cat[cat] += 1
            dataset.append([age, yearly_salary, years_wanted, preffered_customer, res_credit, cat, cat])
            dataset_faults.append([age, yearly_salary, years_wanted, preffered_customer, res_credit, cat, cat])

    print(no_each_cat)

    different = 0
    for credit_fault in faulty_functions:
        for data in dataset:
            res_credit = credit_fault(data[0], data[1], data[2], data[3])
            #determine category:
            cat = 0
            while res_credit > categories[cat]:
                cat += 1

            #saving values that are wrong 
            if cat != data[5]:
                different += 1
                dataset_faults.append([data[0], data[1], data[2], data[3], res_credit, cat, data[6]])

    print(different)
    write_to_file(dataset_faults, file)

main(300, 4, [credit_fault_1, credit_fault_2, credit_fault_3, credit_fault_4], "dataset_generator/credit_dataset/fault_injection/data/dataset.csv")