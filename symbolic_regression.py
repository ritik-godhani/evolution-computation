import random
import operator
import matplotlib.pyplot as plt

# Just to prevent divide by 0 errors
def divide_safe(a, b):
    """
    Divide a by b safely. 
    If b is zero, return 1 instead to avoid errors.
    """
    if b == 0:
        return 1
    else:
        return a / b

# math operations
ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': divide_safe
}

# terminals to use in trees
things = ['x', 'y', -2, -1, 0, 1, 2]

# some parameters
maxDepth = 8
popSize = 30
numGen = 30
mutProb = 0.5
k = 3

# Dataset (from Section B Table 1)
data_points = [
    (-1, -1, -6.33333), (-1, 0, -6), (-1, 1, -5.66667), (-1, 2, -5.33333), (-1, 3, -5), (-1, 4, -4.66667), (-1, 5, -4.33333),
    (0, -1, -4.33333), (0, 0, -4), (0, 1, -3.66667), (0, 2, -3.33333), (0, 3, -3), (0, 4, -2.66667), (0, 5, -2.33333),
    (1, -1, -2.33333), (1, 0, -2), (1, 1, -1.66667), (1, 2, -1.33333), (1, 3, -1), (1, 4, -0.666667), (1, 5, -0.333333),
    (2, -1, -0.333333), (2, 0, 0), (2, 1, 0.333333), (2, 2, 0.666667), (2, 3, 1), (2, 4, 1.33333), (2, 5, 1.66667),
    (3, -1, 1.66667), (3, 0, 2), (3, 1, 2.33333), (3, 2, 2.66667), (3, 3, 3), (3, 4, 3.33333), (3, 5, 3.66667),
    (4, -1, 3.66667), (4, 0, 4), (4, 1, 4.33333), (4, 2, 4.66667), (4, 3, 5), (4, 4, 5.33333), (4, 5, 5.66667),
    (5, -1, 5.66667), (5, 0, 6), (5, 1, 6.33333), (5, 2, 6.66667), (5, 3, 7), (5, 4, 7.33333), (5, 5, 7.66667)
]

# for making nodes
class Node:
    """
    Node of an expression tree. Can be an operator or a terminal.
    """
    def __init__(self, val, l=None, r=None):
        self.val = val
        self.l = l
        self.r = r

    def isLeaf(self):
        """
        Check if node has no children.
        """
        return self.l == None and self.r == None

    def __str__(self):
        """
        Get string form of expression starting at this node.
        """
        if self.isLeaf():
            return str(self.val)
        else:
            return "(" + str(self.l) + " " + str(self.val) + " " + str(self.r) + ")"

# function to make trees
def makeTree(d):
    """
    Recursively build a random expression tree.
    """
    # Base case: return a leaf node
    if d == 0 or (d < maxDepth and random.random() < 0.3):
        value = random.choice(things)
        return Node(value)

    # Recursive case: create an operator node with two subtrees
    operator = random.choice(list(ops.keys()))
    left_subtree = makeTree(d - 1)
    right_subtree = makeTree(d - 1)
    return Node(operator, left_subtree, right_subtree)

def makeTreeRamped(d):
    """
    Recursively builds a random expression tree with varied shapes.
    """
    # Base case: stop growing the tree and return a terminal (variable or constant)
    if d == 0 or random.random() < 0.3:
        value = random.choice(things)
        return Node(value)

    # Recursive case: create an operator node with two randomly built subtrees
    operator = random.choice(list(ops.keys()))
    left_subtree = makeTreeRamped(d - 1)
    right_subtree = makeTreeRamped(d - 1)
    return Node(operator, left_subtree, right_subtree)

# evaluate tree output
def evalTree(n, x, y):
    """
    Calculate the result of the expression tree for given x.
    """
    if n.isLeaf():
        if n.val == 'x':
            return x
        elif n.val == 'y':
            return y
        else:
            return n.val
    left = evalTree(n.l, x, y)
    right = evalTree(n.r, x, y)
    return ops[n.val](left, right)

# calc how good tree is
def fitness(ind, inputs):
    """
    Calculates the mean squared error of a tree on given x values.
    """
    errors = []
    for x, y, target in inputs:
        predicted = evalTree(ind, x, y)
        error = (target - predicted) ** 2
        errors.append(error)
    return sum(errors) / len(errors)


# pick parents
def select(pop, fits):
    """
    Selects two parents using tournament selection.
    """
    sample = random.sample(list(zip(pop, fits)), k)
    sample.sort(key=lambda a: a[1])
    return sample[0][0], sample[1][0]

# swap parts of two trees
def cross(t1, t2):
    """
    Performs crossover between two trees.
    """
    if random.random() > 0.5:
        return t1
    if not t1.isLeaf() and not t2.isLeaf():
        return Node(t1.val, cross(t1.l, t2.l), cross(t1.r, t2.r))
    return random.choice([t1, t2])

# chance to change tree
def change(tree):
    """
    Mutates a tree with some probability.
    """
    if random.random() < mutProb:
        return makeTree(random.randint(1, maxDepth))
    if tree.isLeaf():
        return Node(tree.val)
    return Node(tree.val, change(tree.l), change(tree.r))

# make a population
def popInit(size, maxD):
    """
    Initializes a population of random trees.
    """
    pop = []
    for i in range(size):
        d = random.randint(2, maxD)
        if i < size // 2:
            pop.append(makeTree(d))        # full
        else:
            pop.append(makeTreeRamped(d))  # grow
    return pop

# score all trees
def evalPop(pop, xs):
    """
    Evaluates the fitness of all individuals in the population.
    """
    scores = []
    for p in pop:
        scores.append(fitness(p, xs))
    return scores

# get best one
def best(pop, fits):
    """
    Finds the best individual in the population.
    """
    b = min(fits)
    i = fits.index(b)
    return pop[i], b

# next gen
def nextGen(pop, fits, bestOne):
    """
    Creates the next generation of the population.
    """
    newPop = [bestOne]
    while len(newPop) < popSize:
        p1, p2 = select(pop, fits)
        kid = cross(p1, p2)
        kid = change(kid)
        newPop.append(kid)
    return newPop

# show how good it did
def plotFit(history):
    """
    Plots the fitness progression over generations.
    """
    plt.plot(range(len(history)), history)
    plt.xlabel('Gen')
    plt.ylabel('Best')
    plt.title('Fitness')
    plt.grid(True)
    plt.savefig('fitness.png')
    plt.show()

# compare with real function
def compareFinal(tree, inputs):
    """
    Plots the output of the final evolved tree against the real function.
    """
   
    actual = []
    predicted = []

    for point in inputs:
        x_value = point[0]
        y_value = point[1]
        true_result = point[2]
        actual.append(true_result)
        predicted_result = evalTree(tree, x_value, y_value)
        predicted.append(predicted_result)
    
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel("Data Point Index")
    plt.ylabel("Output")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid()
    plt.savefig("compare.png")
    plt.show()

# main program
def run():
    """
    Runs the genetic programming loop:
    - Initializes population
    - Evolves population over generations
    - Plots fitness and final comparison
    """
    pop = popInit(popSize, maxDepth)
    history = []
    inputs = data_points
    for g in range(numGen):
        fits = evalPop(pop, inputs)
        b, bf = best(pop, fits)
        print("Gen", g, "Best Fitness =", bf)
        history.append(bf)
        if bf < 0.1:
            print("Early stopping: fitness below 0.1")
            break
        pop = nextGen(pop, fits, b)

    fits = evalPop(pop, inputs)
    final, err = best(pop, fits)
    print("\nFinal Expression Tree:")
    print(final)
    print("Final MSE:", err)
    plotFit(history)
    compareFinal(final, inputs)

run()