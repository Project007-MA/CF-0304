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
data_path = 'dataset/Individual.csv'
data = pd.read_csv(data_path)

# Label Encoding
le = LabelEncoder()
le1 = LabelEncoder()
le4 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

data['Mobile_label'] = le.fit_transform(data['Mobile'])
data['AadharNo_label'] = le2.fit_transform(data['AadharNo'])
data['IsGenuine_label'] = le3.fit_transform(data['IsGenuine'])
data['IsRegularBacker_label'] = le4.fit_transform(data['IsRegularBacker'])
# Drop irrelevant columns
df = data.drop(['Name','Mobile','MailID','State','AadharNo','IsGenuine','IsRegularBacker'], axis=1)

# Prepare features and targets
X = df[['AadharNo_label','Mobile_label','IsGenuine_label']]  # Assuming 'rt' is your feature
y = df[['IsRegularBacker_label']]  # Assuming 'er' is your target

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

# Train the BiRNN model and store training history
history = bi_rnn_model.fit(
    X_train_encoded, y_train,
    epochs=50, batch_size=64,
    validation_data=(X_test_encoded, y_test)
)
final_val_accuracy = history.history['val_accuracy'][-1]
print(final_val_accuracy * 100,': Accuracy')
# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.show()
