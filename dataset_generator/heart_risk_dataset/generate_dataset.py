import random
from heart_risk_calculator import heart_risk_calculator

max_heart_risk = 49.39
min_heart_risk = 0

def write_to_file(dataset, file):
    with open(file, "w") as f:
        f.write("gender,age,bmi,exercices,stress,smoking,res_risk,cat\n")
        for data in dataset:
            f.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + "," + str(data[7]) + "\n")

def main(no_cases, no_categories, file):

    #generate categories:
    spred = (max_heart_risk-min_heart_risk)/(no_categories)
    categories = [0]
    for i in range(no_categories):
        categories.append(categories[i]+spred)

    print(categories)

    dataset = []
    no_each_cat = [0] * (no_categories+1)
    while(sum(no_each_cat)<no_cases):
        gender = random.randint(0,1)
        age = random.randrange(20,80)
        bmi = random.randrange(10,200)
        exercices = random.randrange(1,10)
        stress = random.randrange(1,10)
        smoking = random.randint(0,1)
        res_risk = heart_risk_calculator(gender, age, bmi, exercices, stress, smoking)
        #determine category:
        cat = 0
        while res_risk >= categories[cat] and cat < 4:
            cat += 1

        if(no_each_cat[cat]<no_cases/no_categories):
            no_each_cat[cat] += 1
            dataset.append([gender, age, bmi, exercices, stress, smoking, res_risk, cat])

    print(no_each_cat)

    write_to_file(dataset, file)

main(800, 4, "dataset_generator/heart_risk_dataset/data/dataset.csv")