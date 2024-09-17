from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = np.load('model.npz')

W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Updated image preprocessing with better thresholding
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize the image to 28x28
    
    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    return img_array

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Example of prediction on a new image
def predict_image(image_path, W1, b1, W2, b2):
    img_array = load_and_preprocess_image(image_path)
    img_array = img_array.reshape(1, 784).T  # Reshape for prediction (1x784 vector)
    prediction = make_predictions(img_array, W1, b1, W2, b2)
    return prediction, img_array

# Test prediction and visualize
def test_prediction(image_path, W1, b1, W2, b2):
    prediction, current_image = predict_image(image_path, W1, b1, W2, b2)
    label = image_path.split('/')[1]
    
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    # Reshape the image back for visualization
    current_image = current_image.reshape((28, 28)) * 255  # Rescale for visualization
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Example prediction
image_path = 'ejemplos/0.jpeg'  # Replace with your actual image path
test_prediction(image_path, W1, b1, W2, b2)
