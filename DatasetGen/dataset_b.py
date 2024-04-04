import random
import pandas as pd
def generate_datapoints(n):
    data_points = []

    for _ in range(n):
        house_size = int(random.uniform(50, 300))
        num_people = random.randint(1, 8)
        season = random.choice(['summer', 'monsoon', 'winter'])
        Category = calculate_electricity_Category(house_size, num_people, season)
        data_point = {
            'HouseSize': house_size,
            'NumPeople': num_people,
            'Season': season,
            'Category': Category
        }
        data_points.append(data_point)
    return pd.DataFrame(data_points)

def calculate_electricity_Category(house_size, num_people, season):
    size_factor = 3 
    people_factor = 10
    season_factor = {'summer': 1.7, 'monsoon': 1.2, 'winter': 1.1}

    electricity_Category = (
        size_factor * house_size +
        people_factor * num_people +
        10 * season_factor[season]
    )
    if(electricity_Category<300):
        category = 'Tier 3'
    elif(electricity_Category<600):
        category = 'Tier 2'
    else:
        category = 'Tier 1'

    return category
n = 10
generated_data = generate_datapoints(n)
print("Generated dataset")
print(generated_data)
generated_data.to_csv("Categorical.csv")

# print(generated_data.info())

# def predict_electricity_bill():
#     house_size = float(input("Enter house size (in square meters): "))
#     num_people = int(input("Enter number of people: "))
#     season = input("Enter season (summer/monsoon/winter): ")
#     predicted_bill = calculate_electricity_Category(house_size, num_people, season)
#     print(f'Predicted electricity Category: {predicted_bill}')
#predict_electricity_bill()