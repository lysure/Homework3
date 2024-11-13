import numpy as np
import random
import matplotlib.pyplot as plt

# Character labels for reference
char_labels = ["G", "N", "I", "Q", "O", "U", "H", "Z"]

# Font data for each character in six different fonts
font_data = {
    "font1": ["01110100011000110001111111000110001", "11111100001000011110100001000011111",
              "01110001000010000100001000010001110", "11110100011000110001100011000111110",
              "11111100001000011110100001000011111", "10001100011000111111100011000110001",
              "10001100011000110001100011000110001", "11111000010001000100010001000011111"],
    "font2": ["00111010011000110001111111000110001", "11111100001000011110100001000011111",
              "01110001000010000100001000010001110", "11110100011000110001100011000111110",
              "11111100001000011110100001000011111", "10001100011000111111100011000110001",
              "10001100011000110001100011000110001", "11111000010001000100010001000011111"],
    "font3": ["01110010101101110001111111000110001", "11111100001000011110100001000011111",
              "01110001000010000100001000010001110", "11110010100101101001010110101011110",
              "11111100001000011110100001000011111", "10001100011000111111100011000110001",
              "10001100011000110001100011000110001", "11111000010001000100010001000011111"],
    "font4": ["00111010011000110001111111000110001", "11111100001000011110100001000011111",
              "01110001000010000100001000010001110", "11110100011000110001100011000111110",
              "11111100001000011110100001000011111", "10001100011000111111100011000110001",
              "10001100011000110001100011000110001", "11111000010001000100010001000011111"],
    "font5": ["00100010101000110001111111000110001", "11111100001000011110100001000011111",
              "01110001000010000100001000010001110", "11110100011000110001100011000111110",
              "11111100001000011110100001000011111", "10001100011000111111100011000110001",
              "10001100011000110001100011000110001", "11111000010001000100010001000011111"],
    "font6": ["01110100011000111111100011000110001", "11111100001000011110100001000011111",
              "01110001000010000100001000010001110", "11110100011000110001100011000111110",
              "11111100001000011110100001000011111", "10001100011000111111100011000110001",
              "10001100011000110001100011000110001", "11111000010001000100010001000011111"]
}

# Convert font data from binary strings to numeric arrays
def preprocess_font_data(font_data):
    processed_data = []
    labels = []
    for font, characters in font_data.items():
        for i, char_data in enumerate(characters):
            vector = [int(bit) for bit in char_data]  # Convert binary string to list of integers
            processed_data.append(vector)
            labels.append(char_labels[i])  # Use char_labels to maintain consistency with characters
    return np.array(processed_data), labels

# Prepare data
data_matrix, labels = preprocess_font_data(font_data)

# Define SOFM parameters
input_len = 35
output_dim = (5, 5)  # 5x5 output layer
initial_learning_rate = 0.6

# Initialize SOFM weights
W = np.random.rand(25, input_len)

# Define Win-Take-All function
def WTA(x, W):
    dist = np.array([np.dot(x - w, x - w) for w in W])
    return np.argmin(dist)

# Define neighborhood update functions for each topology
def compete0(x, W, eta):
    for xx in x:
        id = WTA(xx, W)
        W[id] = W[id] + eta * (xx - W[id])
    return W

def compete1(x, W, eta):
    for xx in x:
        id = WTA(xx, W)
        W[id] = W[id] + eta * (xx - W[id])
        if id > 0:
            W[id - 1] = W[id - 1] + eta * (xx - W[id - 1])
        if id + 1 < len(W):
            W[id + 1] = W[id + 1] + eta * (xx - W[id + 1])
    return W

def neighbor_ids_2d(id, row, col):
    rown, coln = divmod(id, col)
    ids = [id]
    if coln > 0:
        ids.append(id - 1)
    if coln < col - 1:
        ids.append(id + 1)
    if rown > 0:
        ids.append(id - col)
    if rown < row - 1:
        ids.append(id + col)
    return ids

def compete2(x, W, eta):
    for xx in x:
        id = WTA(xx, W)
        for i in neighbor_ids_2d(id, 5, 5):
            W[i] = W[i] + eta * (xx - W[i])
    return W

# Function to train SOFM
def train_sofm(data, W, compete_func, steps=1000):
    for i in range(steps):
        eta = 0.6 - (0.59 * i / steps)
        data_shuffled = data.copy()
        np.random.shuffle(data_shuffled)
        W = compete_func(data_shuffled, W, eta)
    return W

# Function to display SOFM results
def display_sofm(W, labels, data_matrix):
    plt.figure(figsize=(6, 6))
    label_positions = {i: [] for i in range(len(W))}
    for i, vector in enumerate(data_matrix):
        winner = WTA(vector, W)
        label_positions[winner].append(labels[i])
    for i, pos in label_positions.items():
        plt.text(i % 5, i // 5, ''.join(pos), ha='center', va='center')
    plt.title("SOM Clustering")
    plt.show()

# Train and display results for each topology
print("Standard Competitive Layer (5x5)")
W_standard = train_sofm(data_matrix, W.copy(), compete0)
display_sofm(W_standard, labels, data_matrix)

print("Competitive Layer with Immediate Neighbors (5x5)")
W_neighbors = train_sofm(data_matrix, W.copy(), compete1)
display_sofm(W_neighbors, labels, data_matrix)

print("2D Competitive Layer (5x5 Grid)")
W_2d_topology = train_sofm(data_matrix, W.copy(), compete2)
display_sofm(W_2d_topology, labels, data_matrix)
