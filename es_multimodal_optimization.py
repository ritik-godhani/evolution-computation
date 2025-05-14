import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def rastrigin(X):
    """
    Rastrigin Function - multimodal function used for testing optimization.
    Global minimum is at X=0 with f(X)=0.
    """
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

class EvolutionStrategy:
    """
    Class representing the Evolution Strategy for minimizing a multimodal function.
    """

    def __init__(self, objective_function, dim=2, bounds=(-5.12, 5.12), mu=15, lamb=30, 
                 mutation_strength=0.5, mutation_prob=0.1, generations=100):
        """
        Initialize ES with key parameters.
        """
        self.objective = objective_function  # Optimization objective function
        self.dim = dim # Dimensionality of the solution vector
        self.lower_bound, self.upper_bound = bounds # Search space bounds
        self.mu = mu # Number of parents
        self.lamb = lamb # Number of offspring
        self.sigma = mutation_strength # Standard deviation for Gaussian mutation
        self.mutation_prob = mutation_prob # Probability of mutation per gene
        self.generations = generations # Number of generations to evolve

        # Initialize parent population randomly within bounds
        self.parents = np.random.uniform(self.lower_bound, self.upper_bound, (self.mu, self.dim))
        self.fitness_history = [] # Tracks the best fitness over generations

    def mutate(self, parent):
        """
        Applies mutation to a parent vector based on Gaussian noise.
        """
        child = np.copy(parent)
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                # Apply mutation with Gaussian noise and clip within bounds
                child[i] += np.random.normal(0, self.sigma)
                child[i] = np.clip(child[i], self.lower_bound, self.upper_bound)
        return child

    def evolve(self):
        """
        Evolves the population over generations.
        """
        for gen in range(self.generations):
            offspring = []
            # Generate offspring by mutating random parents
            for _ in range(self.lamb):
                parent_idx = np.random.randint(0, self.mu)
                parent = self.parents[parent_idx]
                child = self.mutate(parent)
                offspring.append(child)

            # Combine parents and offspring into a single population
            combined = np.vstack((self.parents, offspring))
            # Evaluate fitness of all individuals
            fitness = np.array([self.objective(ind) for ind in combined])
            # Select top individuals with the lowest (best) fitness
            best_indices = np.argsort(fitness)[:self.mu]
            self.parents = combined[best_indices]
            # Record the best fitness value of the generation
            best_fitness = fitness[best_indices[0]]
            self.fitness_history.append(best_fitness)

            print(f"Generation {gen + 1}, Best Fitness: {best_fitness:.5f}")

        # Return the best solution found and its fitness history
        return self.parents[0], self.fitness_history

def plot_convergence(fitness_history):
    """
    Plot the convergence of the best fitness value over generations.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(fitness_history, label="Best Fitness")
    plt.title("Convergence Plot")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # Save plot as image
    plt.savefig("convergence_plot.png")
    plt.show()

def plot_landscape_2d(bounds=(-5.12, 5.12), resolution=100):
    """
    Plot 2D contour of the Rastrigin function.
    """
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    # Evaluate Rastrigin function on grid
    Z = np.array([rastrigin([i, j]) for i, j in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    # Generate 2D contour plot
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title("Rastrigin Function Landscape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    # Save plot as image
    plt.savefig("rastrigin_2d_landscape.png")
    plt.show()

def plot_landscape_3d(bounds=(-5.12, 5.12), resolution=100):
    """
    Plot 3D surface of the Rastrigin function.
    """
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    # Evaluate Rastrigin function on grid
    Z = np.array([rastrigin([i, j]) for i, j in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    # Generate 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='none', alpha=0.9)
    ax.set_title("3D Surface of Rastrigin Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.tight_layout()
    # Save plot as image
    plt.savefig("rastrigin_3d_landscape.png")
    plt.show()  

if __name__ == "__main__":
    # Create Evolution Strategy optimizer instance with given parameters
    es = EvolutionStrategy(
        objective_function=rastrigin,
        dim=2,
        bounds=(-5.12, 5.12),
        mu=20,
        lamb=40,
        mutation_strength=0.3,
        mutation_prob=0.1,
        generations=100
    )
    # Run the evolutionary process
    best_solution, fitness_history = es.evolve()

    # Display best solution and its fitness
    print("\nBest solution found:", best_solution)
    print("Best fitness:", rastrigin(best_solution))
    
    # Plot convergence and function landscape
    plot_convergence(fitness_history)
    plot_landscape_2d()
    plot_landscape_3d()