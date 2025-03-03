import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy_loss(y_true, y_pred):

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class NeuralNetwork:
    def __init__(self):
        # Ensure weights and biases are floats to avoid casting issues
        self.weights_input_hidden = np.array([[2.0, -2.0], [2.0, -2.0]], dtype=float)
        self.bias_hidden = np.array([[1.0, -1.0]], dtype=float)
        self.weights_hidden_output = np.array([[2.0], [2.0]], dtype=float)
        self.bias_output = np.array([[-1.0]], dtype=float)

    def forward_propagation(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Hidden layer to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backward_propagation(self, X, y_true):
        # Output error
        output_delta = self.output - y_true
        
        # Hidden delta
        hidden_delta = np.dot(output_delta, self.weights_hidden_output.T)
        

        self.grad_weights_hidden_output = np.dot(self.hidden_output.T, output_delta) 
        self.grad_bias_output = np.sum(output_delta, axis=0, keepdims=True)           
        self.grad_weights_input_hidden = np.dot(X.T, hidden_delta)                  
        self.grad_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)          

    def update_weights(self, learning_rate):
        self.weights_hidden_output -= learning_rate * self.grad_weights_hidden_output
        self.bias_output -= learning_rate * self.grad_bias_output
        self.weights_input_hidden -= learning_rate * self.grad_weights_input_hidden
        self.bias_hidden -= learning_rate * self.grad_bias_hidden


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_true = np.array([
    [0],
    [1],
    [1],
    [0]
])


nn = NeuralNetwork()


print("=== Forward Propagation (Before Update) ===")
output = nn.forward_propagation(X)
cost_before = binary_cross_entropy_loss(y_true, output)
print("Output:\n", output)
print("Cost (binary cross entropy):", cost_before)


nn.backward_propagation(X, y_true)

print("\n=== Gradients (Before Update) ===")
print("grad_weights_hidden_output:\n", nn.grad_weights_hidden_output)
print("grad_bias_output:\n", nn.grad_bias_output)
print("grad_weights_input_hidden:\n", nn.grad_weights_input_hidden)
print("grad_bias_hidden:\n", nn.grad_bias_hidden)


learning_rate = 0.1
nn.update_weights(learning_rate)


print("\n=== Forward Propagation (After Update) ===")
output_after = nn.forward_propagation(X)
cost_after = binary_cross_entropy_loss(y_true, output_after)
print("Output:\n", output_after)
print("Cost (binary cross entropy):", cost_after)


nn.backward_propagation(X, y_true)
print("\n=== Gradients (After Update) ===")
print("grad_weights_hidden_output:\n", nn.grad_weights_hidden_output)
print("grad_bias_output:\n", nn.grad_bias_output)
print("grad_weights_input_hidden:\n", nn.grad_weights_input_hidden)
print("grad_bias_hidden:\n", nn.grad_bias_hidden)