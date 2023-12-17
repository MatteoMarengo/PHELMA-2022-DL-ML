import numpy as np 
#import pickle
import function
#import matplotlib.pyplot as plt

class Network:

    def __init__(self, 
                 num_nodes_in_layers, 
                 batch_size,
                 num_epochs,
                 learning_rate, 
                 weights_file
                 ):

        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights_file = weights_file

        # build the network
        #         w1/b1    w2/b2   
        #784(inputs) ---> 20 ---> 10(output)
        #         x     z1  a1  z2  a2=y
        # Weights and biases initialization (Random for weights and zero for biases)
        self.weight1 = np.random.normal(0, 1, [self.num_nodes_in_layers[0], self.num_nodes_in_layers[1]])
        self.bias1 = np.zeros((1, self.num_nodes_in_layers[1]))
        self.weight2 = np.random.normal(0, 1, [self.num_nodes_in_layers[1], self.num_nodes_in_layers[2]])
        self.bias2 = np.zeros((1, self.num_nodes_in_layers[2]))
        self.loss = []

    def train(self, inputs, labels):

        for epoch in range(self.num_epochs): # training begin
            print('Epoch : ', epoch)
            iteration = 0
            while iteration < len(inputs):
                
                # batch input
                inputs_batch = inputs[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]
                                              
                # forward pass:  output computation
                z1 = np.dot(inputs_batch, self.weight1) + self.bias1
                a1 = function.relu(z1)
                z2 = np.dot(a1, self.weight2) + self.bias2
                y = function.softmax(z2)
                
                # Loss function computation
                loss = function.cross_entropy(y, labels_batch)
                
                #Regularized loss function
                lambda_ = 0 # Default value lambda = 0.01. With lambda = 0 no regularization
                loss += function.L2_regularization(lambda_, self.weight1, self.weight2)#lambda
                self.loss.append(loss)
                                
                # backward pass: gradient computations
                delta_y = (y - labels_batch) / y.shape[0]
                delta_hidden_layer = np.dot(delta_y, self.weight2.T) 
                delta_hidden_layer[a1 <= 0] = 0 # derivatives of relu

                # backpropagation
                weight2_gradient = np.dot(a1.T, delta_y) # forward * backward
                bias2_gradient = np.sum(delta_y, axis = 0, keepdims = True)
            
                weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                bias1_gradient = np.sum(delta_hidden_layer, axis = 0, keepdims = True)

                # L2 regularization
                weight2_gradient += lambda_ * self.weight2
                weight1_gradient += lambda_ * self.weight1

                # stochastic gradient descent with weights and biases updating
                self.weight1 -= self.learning_rate * weight1_gradient #update weight and bias
                self.bias1 -= self.learning_rate * bias1_gradient
                self.weight2 -= self.learning_rate * weight2_gradient
                self.bias2 -= self.learning_rate * bias2_gradient

                #print('=== Epoch: {:d}/{:d}\tIteration:{:d}\tLoss: {:.2f} ===').format(epoch+1, self.num_epochs, iteration+1, loss)
                
                iteration += self.batch_size
                
        #Final output computation
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = function.relu(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probas = function.softmax(scores)
        # Training accuracy computation
        right_answers = np.sum(np.argmax(probas, 1) == np.argmax(labels, 1))
        acc = float(right_answers) / float(len(labels))
        print('Training accuracy: ', round(acc*100, 2))
       

    def test(self, inputs, labels):
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = function.relu(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = function.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        print('Test accuracy: ', round(acc*100, 2))

