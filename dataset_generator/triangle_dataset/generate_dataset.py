import random

# Function definition to check validity
def is_valid_triangle(a,b,c):
    if a+b>=c and b+c>=a and c+a>=b:
        return True
    else:
        return False

# Function definition for type
def type_of_triangle(a,b,c):
    if a==b and b==c:
        return 1
    elif a==b or b==c or a==c:
        return 2
    else:
        return 3
    
def write_to_file(dataset, file):
    with open(file, "w") as f:
        f.write("side_a,side_b,side_c,triangle_type\n")
        for data in dataset:
            f.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "\n")



def main(no_cases, spred, file):
    non_triangle = 0
    equilateral = 0
    isosceles = 0
    scalane = 0 

    dataset=[]

    while equilateral+isosceles+scalane+non_triangle<no_cases:
        side_a = random.randrange(1,200)
        side_b = random.randrange(1,200)
        side_c = random.randrange(1,200)
        if is_valid_triangle(side_a, side_b, side_c):
            type = type_of_triangle(side_a, side_b, side_c)
            if type == 1 and equilateral<no_cases/4+spred:
                dataset.append([side_a, side_b, side_c, 1])
                equilateral+=1
            elif type == 2 and isosceles<no_cases/4+spred:
                dataset.append([side_a, side_b, side_c, 2])
                isosceles+=1
            elif type == 3 and scalane<no_cases/4+spred:
                dataset.append([side_a, side_b, side_c, 3])
                scalane+=1
        elif non_triangle<no_cases/4+spred:
            dataset.append([side_a, side_b, side_c, 4])
            non_triangle+=1

    print('Number of non_triangle: ', non_triangle)
    print('Number of equilateral: ', equilateral)
    print('Number of isosceles: ', isosceles)
    print('Number of scalane: ', scalane)

    write_to_file(dataset, file)
    #print(dataset)

main(800, 5, "dataset_generator/triangle_dataset/data/dataset.csv")