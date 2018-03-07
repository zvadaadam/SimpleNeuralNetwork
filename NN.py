import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(1)

        # We have 3 input neurons with fully connected to one hidden layer 3*4 edges
        # And one output neuron so 4 edges incoming
        self.weights= [2 * np.random.random((3, 4)) - 1, 2 * np.random.random((4, 1)) - 1]

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        func = self.__sigmoid(x)
        return func * (1 - func)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, plot=True):

        print("Begin training")

        training_error = []

        for iteration in range(number_of_training_iterations):

            # Predication set - FEED FORWARD for all layers
            a0 = training_set_inputs

            z1 = np.dot(a0, self.weights[0])
            a1 = self.__sigmoid(z1)

            z2 = np.dot(a1, self.weights[1])
            a2 = self.__sigmoid(z2)

            # Calculate output error
            error_a2 = training_set_outputs - a2
            # Calculate output delta
            delta2 = error_a2 * self.__sigmoid_derivative(z2)

            # Calculate hidden layer error
            error_a1 = np.dot(delta2, self.weights[1].transpose())
            # Calculate hidden layer delta
            delta1 = error_a1 * self.__sigmoid_derivative(z1)

            # print(error_a2)

            # update all layer weights, TODO: add learning rate
            self.weights[0] += np.dot(a0.transpose(), delta1) # why not minus?
            self.weights[1] += np.dot(a1.transpose(), delta2)

            mean_error = np.mean(np.abs(error_a2))
            print("ERROR(" + repr(iteration) + "): " + repr(mean_error))
            training_error.append(mean_error)


        print("Done training")

        if plot:
            plotting_data = {"TrainingError": training_error}

            fig, ax = plt.subplots()
            errors = pd.DataFrame(plotting_data)
            errors.plot(ax=ax)
            plt.show()



if _name_ == "_main_":

    neural_network = NeuralNetwork()

    training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 100000)

    print("Updated weights after training: ")
    print(neural_network.weights[0])
    print(neural_network.weights[1])


