import random
from heart_risk_no_fault import heart_risk_calculator_no_fault
from heart_risk_faulty_1 import heart_risk_calculator_fault_1
from heart_risk_faulty_2 import heart_risk_calculator_fault_2
from heart_risk_faulty_3 import heart_risk_calculator_fault_3
from heart_risk_faulty_4 import heart_risk_calculator_fault_4

max_heart_risk = 49.39
min_heart_risk = 0

def write_to_file(dataset, file):
    with open(file, "w") as f:
        f.write("gender,age,bmi,exercices,stress,smoking,res_risk,cat,actual_cat\n")
        for data in dataset:
            f.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + "," + str(data[7]) + "," + str(data[8]) + "\n")

def main(no_cases, no_categories, faulty_functions, file):

   #generate categories:
    spred = (max_heart_risk-min_heart_risk)/(no_categories)
    categories = [0]
    for i in range(no_categories):
        categories.append(categories[i]+spred)

    print(categories)

    dataset = []
    dataset_faults = []
    no_each_cat = [0] * (no_categories+1)
    while(sum(no_each_cat)<no_cases):
        gender = random.randint(0,1)
        age = random.randrange(20,80)
        bmi = random.randrange(10,200)
        exercices = random.randrange(1,10)
        stress = random.randrange(1,10)
        smoking = random.randint(0,1)
        res_risk = heart_risk_calculator_no_fault(gender, age, bmi, exercices, stress, smoking)
        #determine category:
        cat = 0
        while res_risk >= categories[cat] and cat < 4:
            cat += 1

        if(no_each_cat[cat]<no_cases/no_categories):
            no_each_cat[cat] += 1
            dataset.append([gender, age, bmi, exercices, stress, smoking, res_risk, cat])
            dataset_faults.append([gender, age, bmi, exercices, stress, smoking, res_risk, cat, cat])

    print(no_each_cat)

    different = 0
    for heart_risk_fault in faulty_functions:
        for data in dataset:
            res_risk = heart_risk_fault(data[0], data[1], data[2], data[3], data[4], data[5])
            #determine category:
            cat = 0
            while res_risk > categories[cat] and cat < 4:
                cat += 1

            #saving values that are wrong 
            if cat != data[7]:
                different += 1
                dataset_faults.append([data[0], data[1], data[2], data[3], data[4], data[5], res_risk, cat, data[7]])

    print(different)
    write_to_file(dataset_faults, file)

main(300, 4, [heart_risk_calculator_fault_1, heart_risk_calculator_fault_2, heart_risk_calculator_fault_3, heart_risk_calculator_fault_4], "dataset_generator/heart_risk_dataset/fault_injection/data/dataset.csv")