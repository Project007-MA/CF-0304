import warnings
import numpy as np
import pandas as pd
from keras import Input
from keras.layers import Dense, Bidirectional, SimpleRNN, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

# Load data
data_path = 'dataset/Campaign.csv'
data = pd.read_csv(data_path)
# Label Encoding
le = LabelEncoder()
data['platform_label'] = le.fit_transform(data['Platform'])
data['DurationDays_label'] = le.fit_transform(data['DurationDays'])
data['Backers_label'] = le.fit_transform(data['Backers'])
data['category_label'] = le.fit_transform(data['Category'])
data['IsSuccessful_label'] = le.fit_transform(data['IsSuccessful'])
data['IsReal_label'] = le.fit_transform(data['IsReal'])
data['IsRegularCampaign_label'] = le.fit_transform(data['IsRegularCampaign'])

# Drop irrelevant columns
df = data.drop(['CampaignID', 'CampaignName', 'PlatformURL', 'Platform', 'goalAmount', 'Currency', 'Country', 'Category', 'DurationDays', 'Backers', 'IsSuccessful', 'IsReal', 'IsRegularCampaign'], axis=1)

# Prepare features and targets
X = df[['Backers_label','DurationDays_label','category_label','IsSuccessful_label','IsRegularCampaign_label']]
y = df[['IsReal_label']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=100)

# Define Autoencoder architecture
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.3)(encoded)
decoded = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.3)(decoded)
decoded_output = Dense(X_train.shape[1], activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded_output)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, validation_data=(X_test, X_test))

# Encoder model
encoder_model = Model(inputs=input_layer, outputs=encoded)
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)
X_train_encoded = np.reshape(X_train_encoded, (X_train_encoded.shape[0], 1, X_train_encoded.shape[1]))
X_test_encoded = np.reshape(X_test_encoded, (X_test_encoded.shape[0], 1, X_test_encoded.shape[1]))

# Define BiRNN model
bi_rnn_input = Input(shape=(1, X_train_encoded.shape[2]))
bi_rnn = Bidirectional(SimpleRNN(512, return_sequences=False, kernel_regularizer=l2(0.001)))(bi_rnn_input)
bi_rnn = Dropout(0.5)(bi_rnn)
dense_layer = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(bi_rnn)
dense_layer = BatchNormalization()(dense_layer)
dense_layer = Dropout(0.5)(dense_layer)
output = Dense(1, activation='sigmoid')(dense_layer)

bi_rnn_model = Model(inputs=bi_rnn_input, outputs=output)
bi_rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# Train BiRNN model
history = bi_rnn_model.fit(X_train_encoded, y_train, epochs=20, batch_size=64, validation_data=(X_test_encoded, y_test))
final_val_accuracy = history.history['val_accuracy'][-1]
print('Accuracy :',final_val_accuracy)
# Plot training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

