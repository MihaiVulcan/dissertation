# gender(0 or 1) age(20, 80), bmi(10, 200), exercices(1,10), stress(1, 10), smoking (0 ro 1)
def heart_risk_calculator_fault_4 (gender, age, bmi, exercices, stress, smoking):
    risk = 0
    if gender == 0:
        if age < 30:
            risk = risk
        elif age > 50: # > insted of <
            risk = risk + 1 
        elif age < 65:
            risk = risk + 3
        elif age <= 80:
            risk = risk + 5

        if bmi < 20:
            risk = risk
        elif bmi < 30:
            risk = risk + 1
        elif bmi < 70:
            risk = risk + 4
        elif bmi <= 100:
            risk = risk + 6
        elif bmi <= 200:
            risk = risk + 10

    elif gender == 1:
        if age < 30:
            risk = risk
        elif age > 50: # > insted of <
            risk = risk + 1
        elif age < 65:
            risk = risk + 4
        elif age <= 80:
            risk = risk + 6

        if bmi < 20:
            risk = risk
        elif bmi < 30:
            risk = risk + 1
        elif bmi < 70:
            risk = risk + 5
        elif bmi <= 100:
            risk = risk + 7
        elif bmi <= 200:
            risk = risk + 12
    
    if exercices < 3:
        risk = risk * 1.4
    elif exercices < 7:
        risk = risk * 1.2
    else:
        risk = risk

    if stress < 3:
        risk = risk
    elif stress < 7:
        risk = risk * 1.2
    else:
        risk = risk * 1.4

    if smoking == 1:
        risk = risk * 1.4

    return risk
