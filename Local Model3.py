import warnings
import numpy as np
import pandas as pd
from keras import Input
from keras.layers import Dense, Bidirectional, SimpleRNN
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load data
data_path = 'dataset/Organization.csv'
data = pd.read_csv(data_path)

# Label Encoding
le = LabelEncoder()
le1 = LabelEncoder()
le4 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

data['OrgName_label'] = le.fit_transform(data['OrgName'])
data['Sector_label'] = le2.fit_transform(data['Sector'])
data['IsGenuine_label'] = le3.fit_transform(data['IsGenuine'])
data['IsRegularDonor_label'] = le4.fit_transform(data['IsRegularDonor'])
# Drop irrelevant columns
df = data.drop(['OrgName','Founded','HQLocation','Sector','NetWorthInMillions(USD)','Website','IsGenuine','IsRegularDonor'], axis=1)

# Prepare features and targets
X = df[['OrgName_label','Sector_label','IsRegularDonor_label']]  # Assuming 'rt' is your feature
y = df[['IsGenuine_label']]  # Assuming 'er' is your target

# Ensure target variable is binary (0 or 1)
y = y.applymap(lambda x: 1 if x > 0.5 else 0)  # Adjust threshold if needed

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=100)

# Define Autoencoder architecture
input_layer = Input(shape=(X_train.shape[1],))
# Encoder
encoded = Dense(512, activation='relu')(input_layer)
# Decoder
decoded = Dense(512, activation='relu')(encoded)
# Output of the Autoencoder
decoded_output = Dense(X_train.shape[1], activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded_output)

# Compile the Autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_data=(X_test, X_test))

# Encoder model
encoder_model = Model(inputs=input_layer, outputs=encoded)

# Get encoded features for the BiRNN input
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)

# Reshape encoded features for RNN input (RNNs expect 3D input: batch_size, timesteps, features)
X_train_encoded = np.reshape(X_train_encoded, (X_train_encoded.shape[0], 1, X_train_encoded.shape[1]))
X_test_encoded = np.reshape(X_test_encoded, (X_test_encoded.shape[0], 1, X_test_encoded.shape[1]))

# Define the BiRNN model for classification
bi_rnn_input = Input(shape=(1, X_train_encoded.shape[2]))
bi_rnn = Bidirectional(SimpleRNN(512, return_sequences=False))(bi_rnn_input)

# Add Dense layers for classification
dense_layer = Dense(512, activation='relu')(bi_rnn)
output = Dense(1, activation='sigmoid')(dense_layer)

# BiRNN model
bi_rnn_model = Model(inputs=bi_rnn_input, outputs=output)

# Compile the BiRNN model
bi_rnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  # Changed to binary_crossentropy

# Print the BiRNN model summary
print(bi_rnn_model.summary())

# Train the BiRNN model
history_birnn = bi_rnn_model.fit(X_train_encoded, y_train, epochs=50, batch_size=64, validation_data=(X_test_encoded, y_test))

# Predict the test set results
y_pred = bi_rnn_model.predict(X_test_encoded)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary (0 or 1)

# Flatten y_test and y_pred to ensure they are 1D arrays
y_test_flat = y_test.values.flatten()
y_pred_flat = y_pred.flatten()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)

# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot accuracy and loss for Autoencoder and BiRNN
plt.figure(figsize=(12, 5))

# Autoencoder loss
plt.subplot(1, 2, 1)
plt.plot(history_autoencoder.history['loss'], label='Train Loss')
plt.plot(history_autoencoder.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# BiRNN accuracy
plt.subplot(1, 2, 2)
plt.plot(history_birnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_birnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('BiRNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
