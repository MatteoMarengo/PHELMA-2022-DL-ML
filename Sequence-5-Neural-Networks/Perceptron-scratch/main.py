import numpy as np
from network import Network
import mnist
import time


# load data
num_classes = 10
train_images = mnist.train_images() #[60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print("Training...")

# data processing
X_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_train = (X_train / 255) #Image normalization
y_train = np.eye(num_classes)[train_labels] #convert label to one-hot encoded vector

X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_test = (X_test / 255) #normalization
y_test = test_labels

# Network architecture and hyperparameters
net = Network(
                 num_nodes_in_layers = [784, 5, 10], 
                 batch_size = 200,
                 num_epochs = 15,
                 learning_rate = 0.5, 
                 weights_file = 'weights.pkl'
             )
# Network training
start_time = time.time()
net.train(x_train, y_train)
end_time = time.time()
print('Training duration in s:', round(end_time - start_time, 2))

# Network performances on the test set
net.test(x_test, y_test)
