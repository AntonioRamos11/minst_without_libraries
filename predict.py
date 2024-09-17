from PIL import Image
import numpy as np

# Load the model
model = np.load('model.npz')

W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

# Updated image preprocessing with background removal
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize the image to 28x28
    img_array = np.array(img)  # Convert the image to a numpy array
    
    # Apply a threshold to remove the background
    threshold = 200  # Adjust this threshold value based on your images

    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    
    return img_array

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Example of prediction on a new image
def predict_image(image_path, W1, b1, W2, b2):
    img_array = load_and_preprocess_image(image_path)
    img_array = img_array.reshape(1, 784).T  # Reshape for prediction
    prediction = make_predictions(img_array, W1, b1, W2, b2)
    return prediction, img_array

# Predicting a new image
image_path = 'ejemplos/0.jpeg'  # Replace with your image path
prediction = predict_image(image_path, W1, b1, W2, b2)
print(f'Predicted Class for the image: {prediction[0]}')

# Test prediction and visualize
import matplotlib.pyplot as plt

def test_prediction(image_path, W1, b1, W2, b2):
    prediction, current_image = predict_image(image_path, W1, b1, W2, b2)
    label = image_path.split('/')[1]
    
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

image_path = 'ejemplos/0.jpeg'  # Replace with your image path
test_prediction(image_path, W1, b1, W2, b2)
