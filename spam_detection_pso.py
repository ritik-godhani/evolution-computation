import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# Load and preprocess dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Vectorize the text field which contains message content
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

# Baseline model
def train_baseline_model(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01, max_iter=200, solver='adam', random_state=42)
    # Train the baseline MLP Classifier NN Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Generate the confusion matrix from Y predicted and Y actual from the test set
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("baseline_model_confusion_matrix.png")
    plt.show()

    return accuracy_score(y_test, y_pred)

# Random Search optimization
def random_search_optimization(X_train, y_train, X_test, y_test):
    param_dist = {'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50)], 'learning_rate_init': [0.001, 0.01, 0.1]}
    model = MLPClassifier(max_iter=200, solver='adam', random_state=42)
    # Search for the best parameters for our Base Model using Random Search Optimizer
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Generate the confusion matrix from Y predicted and Y actual from the test set
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.savefig("random_search_confusion_matrix.png")
    plt.show()
    return accuracy_score(y_test, y_pred), random_search.best_params_

# PSO Optimization
def fitness_function(position, X_train, y_train, X_test, y_test):
    lr, hidden = position[0], int(position[1])
    hidden = max(1, hidden)  # Ensure at least 1 neuron
    model = MLPClassifier(hidden_layer_sizes=(hidden,), learning_rate_init=lr, max_iter=200, solver='adam', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return 1 - accuracy_score(y_test, y_pred)  # Minimize error

class Particle:
    def __init__(self, bounds, X_train, y_train, X_test, y_test):
        self.position = [random.uniform(bounds[0][0], bounds[0][1]), random.uniform(bounds[1][0], bounds[1][1])]
        self.velocity = [random.uniform(-1, 1) for _ in range(2)]
        self.fitness = fitness_function(self.position, X_train, y_train, X_test, y_test)
        self.best_position = list(self.position)
        self.best_fitness = self.fitness

    def update_velocity(self, global_best, w, c1, c2):
        for i in range(2):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds):
        for i in range(2):
            self.position[i] += self.velocity[i]
            self.position[i] = max(bounds[i][0], min(bounds[i][1], self.position[i]))

# PSO Algorithm
def pso_optimization(X_train, y_train, X_test, y_test, swarm_size=10, generations=20, w=0.729, c1=1.49, c2=1.49):
    bounds = [(0.0001, 0.1), (5, 100)]
    swarm = [Particle(bounds, X_train, y_train, X_test, y_test) for _ in range(swarm_size)]
    global_best = min(swarm, key=lambda p: p.best_fitness)
    fitness_history = []
    # Iterate and get the best parameters with minimal error
    for _ in range(generations):
        for particle in swarm:
            particle.update_velocity(global_best.best_position, w, c1, c2)
            particle.update_position(bounds)
            particle.fitness = fitness_function(particle.position, X_train, y_train, X_test, y_test)
            if particle.fitness < particle.best_fitness:
                particle.best_fitness, particle.best_position = particle.fitness, list(particle.position)
        
        global_best = min(swarm, key=lambda p: p.best_fitness)
        fitness_history.append(1 - global_best.best_fitness)
    
    best_lr, best_hidden = global_best.best_position
    best_hidden = int(max(1, best_hidden))
    final_model = MLPClassifier(hidden_layer_sizes=(best_hidden,), learning_rate_init=best_lr, max_iter=200, solver='adam', random_state=42)
    final_model.fit(X_train, y_train)
    # Plot and save confusion matrix of our best model
    ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test)
    plt.title("PSO Optimized Model - Confusion Matrix")
    plt.savefig("pso_confusion_matrix.png")
    plt.show()

    return 1 - global_best.best_fitness, global_best.best_position, fitness_history

# Visualization
def plot_results(fitness_history):
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, marker='o', linestyle='-', label='PSO Optimization')
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("PSO Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pso_fitness_history.png")
    plt.show()

if __name__ == "__main__":
    filepath = "balanced_spam_data.csv"  # Update with your local path
    (X_train, X_test, y_train, y_test), vectorizer = load_data(filepath)

    print("Training Baseline Model...")
    start_time = datetime.now()
    baseline_acc = train_baseline_model(X_train, y_train, X_test, y_test)
    end_time = datetime.now()
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Time taken for training baseline model : {(end_time-start_time).seconds} seconds\n")

    print("Performing Random Search Hyperparameter Tuning...")
    start_time = datetime.now()
    rand_acc, best_params = random_search_optimization(X_train, y_train, X_test, y_test)
    end_time = datetime.now()
    print(f"Random Search Accuracy: {rand_acc:.4f}, Best Params: {best_params}")
    print(f"Time taken for Random Search on baseline model : {(end_time-start_time).seconds} seconds\n")

    print("Optimizing with PSO...")
    start_time = datetime.now()
    # Stick to default parameters for best optimization
    pso_acc, best_pso_params, fitness_history = pso_optimization(X_train, y_train, X_test, y_test)

    ## Parameters for fast search
    pso_acc, best_pso_params, fitness_history = pso_optimization(X_train, y_train, X_test, y_test, swarm_size=5, generations=5, w=0.6, c1=1.2, c2=1.2)
    end_time = datetime.now()
    print(f"PSO Accuracy: {pso_acc:.4f}, Best PSO Params: {best_pso_params}")
    print(f"Time taken for PSO Optimization : {(end_time-start_time).seconds} seconds\n")

    # Plot results
    plot_results(fitness_history)
    
    # Compare results
    methods = ['Baseline', 'Random Search', 'PSO']
    accuracies = [baseline_acc, rand_acc, pso_acc]
    sns.barplot(x=methods, y=accuracies)
    plt.title("Comparison of Different Optimization Techniques")
    plt.ylabel("Accuracy")
    # plt.savefig("optimization_comparison.png")
    plt.show()