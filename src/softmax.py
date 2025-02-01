import numpy as np

# TODO: fix
def softmax(logits):
    exp_logits = np.exp(logits)  
    return exp_logits

# Cross-entropy loss
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))  # Adding small value for stability

def main():
    # Example input: a flattened 28x28 grayscale image (MNIST-like)
    X = np.array([0.2, 0.5, 0.8])  # Simplified feature vector

    # Example weight matrix (3 input features, 3 output classes)
    W = np.array([
        [0.5, -0.2, 0.3],
        [0.1, 0.7, -0.5],
        [-0.3, 0.2, 0.8]
    ])

    # Bias terms for each class
    b = np.array([0.1, -0.1, 0.2])

    # Output Example
    y_true = np.array([1, 0, 0])  # True class: Cat

    # Compute raw scores (logits)
    logits = np.dot(X, W) + b

    print("Raw Scores (Logits):", logits)

    y_pred = softmax(logits)
    print("Softmax Output:", y_pred)

    loss = cross_entropy(y_true, y_pred)

    print("Cross-Entropy Loss:", loss)

if __name__ == "__main__":
    main()