import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, GlobalAveragePooling1D
import shap
from scipy.special import expit
import matplotlib.pyplot as plt

# Load and preprocess data
raw_csv_data = pd.read_csv("ESTC Denorm.csv")  # Update with your actual file path
df = raw_csv_data.copy()
x = df.values
df = pd.DataFrame(x)

# Split data into train and test sets
split = int(df.shape[0] * 0.8)
train_dataset = df.loc[:split, :]
test_dataset = df.loc[split:, :]

# Extract variables
train_xi = train_dataset.loc[:, 1:14]  # Time-independent variables
test_xi = test_dataset.loc[:, 1:14]
train_xg = train_dataset.loc[:, 15:]   # Time-dependent variables
test_xg = test_dataset.loc[:, 15:]

# Reshape time-dependent data
train_xd = train_xg.values.reshape(train_xg.shape[0], 3, 5)  # (samples, timesteps, features)
test_xd = test_xg.values.reshape(test_xg.shape[0], 3, 5)

train_y = train_dataset.iloc[:, 0]
test_y = test_dataset.iloc[:, 0]

# Define the model
transformer_input = Input(shape=(3, 5), name='Transformer_input')
dnn_input = Input(shape=(14,), name='NN_input')

# Process Transformer input and reduce it to a 1D vector
transformer_output = Dense(5, activation='relu')(transformer_input)  # Example layer
transformer_output = GlobalAveragePooling1D()(transformer_output)  # Shape: (None, 5)

# Process DNN input
dnn_dense = Dense(10, activation='relu')(dnn_input)  # Shape: (None, 10)

# Concatenate the two 1D vectors
concat = concatenate([transformer_output, dnn_dense], name='Concatenate')  # Shape: (None, 15)

# Final output layer
final_output = Dense(1, activation='sigmoid')(concat)
final_model = Model(inputs=[transformer_input, dnn_input], outputs=final_output)

# Load pre-trained weights (update with your actual weights file)
final_model.load_weights('best_weights.hdf5')

# Define a model up to the Concatenate layer for SHAP analysis
concat_model = Model(inputs=final_model.input, outputs=final_model.get_layer('Concatenate').output)

# Define prediction function for SHAP
def model_predict(inputs):
    xd_flat = inputs[:, :15]
    xi = inputs[:, 15:]
    xd_reshaped = xd_flat.reshape(-1, 3, 5)
    concat_output = concat_model.predict([xd_reshaped, xi], verbose=0)
    # Replace 'bestsol1' with your actual weights (example provided)
    bestsol1 = np.ones(15)  # Placeholder; update with your actual weights
    predictions = expit(np.dot(concat_output, bestsol1))
    return predictions

# Prepare data for SHAP
train_xd_flat = train_xd.reshape(train_xd.shape[0], -1)
test_xd_flat = test_xd.reshape(test_xd.shape[0], -1)
background_data = np.hstack([train_xd_flat, train_xi])[:100]
test_data = np.hstack([test_xd_flat, test_xi])

# Create SHAP explainer and compute SHAP values
explainer = shap.KernelExplainer(model_predict, background_data)
shap_values = explainer.shap_values(test_data[:10], nsamples=100)

# Visualize SHAP values
feature_names = [f'xd_{i}' for i in range(15)] + [f'xi_{i}' for i in range(14)]
shap.summary_plot(shap_values, test_data[:10], feature_names=feature_names)
plt.savefig('shap_summary.png')