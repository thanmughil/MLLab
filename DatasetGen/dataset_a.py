import random
import pandas as pd
def generate_datapoints(n):
    data_points = []

    for _ in range(n):
        house_size = int(random.uniform(50, 300))
        num_people = random.randint(1, 8)
        season = random.choice(['summer', 'monsoon', 'winter'])
        electricity_bill = calculate_electricity_usage(house_size, num_people, season)
        data_point = {
            'HouseSize': house_size,
            'NumPeople': num_people,
            'Season': season,
            'Usage': electricity_bill
        }
        data_points.append(data_point)
    return pd.DataFrame(data_points)

def calculate_electricity_usage(house_size, num_people, season):
    size_factor = 3 
    people_factor = 15
    season_factor = {'summer': 1.6, 'monsoon': 1.2, 'winter': 1.1}

    electricity_usage = (
        size_factor * house_size +
        people_factor * num_people +
        10 * season_factor[season]
    )

    return round(electricity_usage,2)

n = 10
generated_data = generate_datapoints(n)
print("Generated dataset")
print(generated_data)
generated_data.to_csv("Continuous.csv")

def predict_electricity_bill():
    house_size = float(input("Enter house size (in square meters): "))
    num_people = int(input("Enter number of people: "))
    season = input("Enter season (summer/monsoon/winter): ")
    predicted_bill = calculate_electricity_usage(house_size, num_people, season)
    print(f'Predicted electricity bill for the entered data: {predicted_bill}')

predict_electricity_bill()
