#age (20, 65), yearly_salary(10000, 1000000), year_wanted(1, 30), preffed_custommer{0, 1}
def credit_fault_1(age, yearly_salary, years_wanted, preffed_custommer):
    max_pay_year = 0
    if preffed_custommer == 0:
        monthly_salary = yearly_salary / 16 # 15 insted of 12
        min_month_salary = 0
        if age < 30:
            min_month_salary = monthly_salary*0.9
        elif age < 50:
            min_month_salary = monthly_salary*0.8
        elif age <= 65:
            min_month_salary = monthly_salary*0.7

        max_pay_month = 0
        if yearly_salary < 25000:
            max_pay_month = 0.5 * min_month_salary
        elif yearly_salary >= 25000 and yearly_salary < 50000:
            max_pay_month = 0.6 * min_month_salary
        elif yearly_salary >= 50000 and yearly_salary < 75000:
            max_pay_month = 0.7 * min_month_salary
        elif yearly_salary >= 75000 and yearly_salary <= 100000:
             max_pay_month = 0.8 * min_month_salary

        max_pay_year = max_pay_month * 12

    elif preffed_custommer == 1:
        monthly_salary = yearly_salary / 16 # 15 insted of 12
        if age < 30:
            min_month_salary = monthly_salary*0.9
        elif age < 50:
            min_month_salary = monthly_salary*0.8
        elif age <= 65:
            min_month_salary = monthly_salary*0.7

        max_pay_month = 0
        if yearly_salary < 25000:
            max_pay_month = 0.6 * min_month_salary
        elif yearly_salary >= 25000 and yearly_salary < 50000:
            max_pay_month = 0.7 * min_month_salary
        elif yearly_salary >= 50000 and yearly_salary < 75000: 
            max_pay_month = 0.8 * min_month_salary
        elif yearly_salary >= 75000 and yearly_salary <= 100000:
             max_pay_month = 0.9 * min_month_salary

        max_pay_year = max_pay_month * 12
        
    max_loan = max_pay_year*years_wanted
    return round(max_loan)