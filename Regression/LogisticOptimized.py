import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("../csv/URL_data.csv")

X = df.drop(columns=['target'])
y = df['target']

def evaluate_logistic_regression(selected_features, X_train, X_test, y_train, y_test):
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    model = LogisticRegression()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1

def fitness_function(selected_features):
    accuracy, _, _, _ = evaluate_logistic_regression(selected_features, X_train, X_test, y_train, y_test)
    return accuracy

def mayfly_optimization_algorithm(num_features, num_iterations, population_size):
    population = np.random.randint(2, size=(population_size, num_features))
    best_solution = None
    best_fitness = -1
    
    for iteration in range(num_iterations):
        print(f'Iteration : {iteration}')
        fitness_values = np.array([fitness_function(individual) for individual in population])
        
        max_fitness_index = np.argmax(fitness_values)
        if fitness_values[max_fitness_index] > best_fitness:
            best_fitness = fitness_values[max_fitness_index]
            best_solution = population[max_fitness_index]
        
        probabilities = fitness_values / np.sum(fitness_values)
        selected_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
        selected_population = population[selected_indices]
        
        crossover_point = np.random.randint(num_features)
        crossover_mask = np.arange(num_features) < crossover_point
        offspring_population = selected_population[np.random.permutation(population_size)].reshape(-1, num_features)
        offspring_population[:, crossover_mask] = selected_population[:, crossover_mask]
        
        mutation_rate = 0.01
        mutation_mask = np.random.rand(population_size, num_features) < mutation_rate
        offspring_population[mutation_mask] = 1 - offspring_population[mutation_mask]
        
        population = offspring_population
    
    return best_solution, best_fitness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.shape[1]
num_iterations = 10
population_size = 50

best_solution, best_fitness = mayfly_optimization_algorithm(num_features, num_iterations, population_size)

accuracy, precision, recall, f1 = evaluate_logistic_regression(best_solution, X_train, X_test, y_train, y_test)

print("Best feature subset:", best_solution)
print("Best fitness (accuracy):", best_fitness)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)