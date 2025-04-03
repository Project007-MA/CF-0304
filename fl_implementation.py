import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt  # Import matplotlib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from fl_implementation_utils import *  # Ensure this utility is correctly imported

# Set the default font to Times New Roman and make it bold
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 27
# Define the paths to the datasets
data_paths = [
    r'E:\Blockchain-SmartHome\sajin\Federated-Learning\data\Campaign.csv',
    r'E:\Blockchain-SmartHome\sajin\Federated-Learning\data\Individual.csv',
    r'E:\Blockchain-SmartHome\sajin\Federated-Learning\data\Organization.csv',
]

def load_data(path):
    try:
        df = pd.read_csv(path)
        print("Columns in the DataFrame:", df.columns)  # Debug: Print column names

        # Adjust the column name based on the dataset
        if 'Class' in df.columns:
            target_column = 'Class'
        elif 'IsSuccessful_label' in df.columns:
            target_column = 'IsSuccessful_label'
        elif 'IsGenuine_label' in df.columns:
            target_column = 'IsGenuine_label'
        else:
            raise KeyError("The target column is not found in the DataFrame.")

        data = df.drop(columns=target_column).values
        labels = df[target_column].values

        return data, labels
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except KeyError as e:
        print(f"Error: {e}")
        return None, None

# Load and combine data
data_list = []
label_list = []

for path in data_paths:
    data, labels = load_data(path)
    if data is not None and labels is not None:
        data_list.append(data)
        label_list.append(labels)

# Check if data_list and label_list are not empty
if not data_list or not label_list:
    raise ValueError("No data loaded. Please check the dataset and column names.")

# Debugging print statements to check feature dimensions
for i, data in enumerate(data_list):
    print(f"Dataset {i} shape: {data.shape}")

# Define a function to align features
def align_features(data_list):
    max_features = max(data.shape[1] for data in data_list)
    aligned_data_list = []
    for data in data_list:
        aligned_data = np.zeros((data.shape[0], max_features))
        aligned_data[:, :data.shape[1]] = data
        aligned_data_list.append(aligned_data)
    return aligned_data_list

# Align feature dimensions
data_list = align_features(data_list)

# Combine all data and labels into single arrays
data_list = np.concatenate(data_list, axis=0)
label_list = np.concatenate(label_list, axis=0)

# Ensure labels are integers for proper processing
label_list = np.asarray(label_list, dtype=int)

# Binarize the labels
n_values = np.max(label_list) + 1
label_list = np.eye(n_values)[label_list]

# Debugging prints
print("Data shape:", data_list.shape)
print("Labels shape:", label_list.shape)
print("Unique labels:", np.unique(label_list, axis=0))

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.1, random_state=42)

# Create clients
clients = create_clients(X_train, y_train, num_clients=10, initial='client')

# Process and batch the training data for each client
clients_batched = {client_name: batch_data(data) for client_name, data in clients.items()}

# Process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

# Define training parameters
comms_round = 10
lr = 0.001 
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# Define BI-RNN class
class BiRNN:
    def build(self, input_dim, output_dim):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(input_dim, 1)))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        return model

# Initialize global model with Adam optimizer
bi_rnn_global = BiRNN()
global_model = bi_rnn_global.build(X_train.shape[1], len(np.unique(y_train, axis=0)))

# Reshape input data for RNN
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

optimizer = Adam(learning_rate=lr)
global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Lists to store accuracy and loss for plotting
global_accuracy = []
global_loss = []

# Commence global training loop
for comm_round in range(comms_round):
    
    # Get the global model's weights
    global_weights = global_model.get_weights()
    
    # List to collect local model weights after scaling
    scaled_local_weight_list = []

    # Randomize client data - using keys
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)
    
    # Loop through each client and create a new local model
    for client in tqdm(client_names, desc='Progress Bar'):
        bi_rnn_local = BiRNN()
        local_model = bi_rnn_local.build(X_train.shape[1], len(np.unique(y_train, axis=0)))
        
        local_optimizer = Adam(learning_rate=lr)
        local_model.compile(loss=loss, optimizer=local_optimizer, metrics=metrics)
        
        local_model.set_weights(global_weights)
        local_model.fit(clients_batched[client], epochs=5, verbose=0)  # Increased epochs
        
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        K.clear_session()
    
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    global_model.set_weights(average_weights)

    epoch_accuracy = []
    epoch_loss = []
    for X_test_batch, Y_test_batch in test_batched:
        global_acc, global_loss_value = test_model(X_test_batch, Y_test_batch, global_model, comm_round)
        epoch_accuracy.append(global_acc)
        epoch_loss.append(global_loss_value)

    avg_acc = np.mean(epoch_accuracy)
    avg_loss = np.mean(epoch_loss)
    
    global_accuracy.append(avg_acc)
    global_loss.append(avg_loss)
    
    print(f"Communication Round {comm_round + 1} - Global Accuracy: {avg_acc}, Global Loss: {avg_loss}")

# Define and train BI-RNN model
bi_rnn_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train)).shuffle(len(y_train)).batch(320)
bi_rnn_model = BiRNN().build(X_train.shape[1], len(np.unique(y_train, axis=0))) 

# Compile the BI-RNN model
bi_rnn_model.compile(loss=loss, 
                     optimizer=Adam(learning_rate=lr), 
                     metrics=metrics)

# Fit the BI-RNN training data to the model
history = bi_rnn_model.fit(bi_rnn_dataset, epochs=100, verbose=0, validation_data=(X_test_reshaped, y_test))

# Plot accuracy and loss for Federated Learning
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, comms_round + 1), global_accuracy, marker='o', label='Global Accuracy')
plt.xlabel('Communication Round',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
# plt.title('Federated Learning - Accuracy', fontsize=14, fontweight='bold')
plt.legend(loc='best', prop={'weight': 'bold'})

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(range(1, comms_round + 1), global_loss, marker='o', label='Global Loss')
plt.xlabel('Communication Round',fontweight='bold')
plt.ylabel('Loss',fontweight='bold')
# plt.title('Federated Learning - Loss', fontsize=14, fontweight='bold')
plt.legend(loc='best', prop={'weight': 'bold'})

plt.tight_layout()
plt.show()

# Plot accuracy and loss for BI-RNN model
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
# plt.title('BI-RNN Global Model - Accuracy', fontsize=14, fontweight='bold')
plt.legend(loc='best', prop={'weight': 'bold'})

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
# plt.title('BI-RNN Global Model - Loss', fontsize=14, fontweight='bold')
plt.legend(loc='best', prop={'weight': 'bold'})

plt.tight_layout()
plt.show()


def plot_confusion_matrix(y_true, y_pred, title, labels):
    """
    Plot a confusion matrix.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    title (str): Title of the plot.
    labels (list): List of labels to be used for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Labels',fontweight='bold')
    plt.ylabel('True Labels', fontweight='bold')
    plt.xticks(family='Times New Roman', fontweight='bold')
    plt.yticks(family='Times New Roman', fontweight='bold')
    plt.show()

def evaluate_global_model(X_test, y_test, model):
    """
    Evaluate and plot confusion matrix for the Global Model.

    Parameters:
    X_test (array-like): Test features.
    y_test (array-like): True labels.
    model (Keras model): Trained model.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    labels = ['Not Genuine', 'Genuine']
    plot_confusion_matrix(y_true, y_pred, 'Global Model Confusion Matrix', labels)

def evaluate_bi_rnn_model(X_test_reshaped, y_test, model):
    """
    Evaluate and plot confusion matrix for the BI-RNN Model.

    Parameters:
    X_test_reshaped (array-like): Test features reshaped for BI-RNN.
    y_test (array-like): True labels.
    model (Keras model): Trained BI-RNN model.
    """
    y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
    y_true = np.argmax(y_test, axis=1)
    labels = ['Not Genuine', 'Genuine']
    plot_confusion_matrix(y_true, y_pred, 'BI-RNN Global Model Confusion Matrix', labels)

# Example usage
# Assume X_test, y_test, global_model, X_test_reshaped, bi_rnn_model are already defined
evaluate_global_model(X_test, y_test, global_model)
evaluate_bi_rnn_model(X_test_reshaped, y_test, bi_rnn_model)
