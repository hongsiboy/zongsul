import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import Layer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr
from scipy.special import expit
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom Positional Encoding Layer
class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Block
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

# Load and preprocess data
raw_csv_data = pd.read_csv(r"ESTC Denorm.csv")
df = raw_csv_data.copy()
x = df.values
df = pd.DataFrame(x)

split = int(df.shape[0] * 0.8)
train_dataset = df.loc[:split, :]
test_dataset = df.loc[split:, :]

train_xi = train_dataset.loc[:, 1:14]  # Time-independent variables for NN
test_xi = test_dataset.loc[:, 1:14]
train_xg = train_dataset.loc[:, 15:]   # Time-dependent variables for Transformer
test_xg = test_dataset.loc[:, 15:]

train_xd = train_xg.values.reshape(train_xg.shape[0], 3, 5)  # (samples, timesteps, features)
test_xd = test_xg.values.reshape(test_xg.shape[0], 3,  CO5)

train_y = train_dataset.iloc[:, 0]
test_y = test_dataset.iloc[:, 0]

# OMA Optimization for Model Architecture
def objfun(x):
    nn_neurons = int(x[0])
    transformer_head_size = int(x[1])
    transformer_num_heads = int(x[2])
    transformer_ff_dim = int(x[3])
    nn_dense_neurons = int(x[4])
    transformer_dropout = x[5]
    nn_dropout_rate = x[6]
    lr = x[7]

    # Ensure head_size is a divisor of d_model (64)
    possible_head_sizes = [2, 4, 8, 16, 32, 64]
    transformer_head_size = min(possible_head_sizes, key=lambda k: abs(k - transformer_head_size))

    # Transformer for time-dependent data
    transformer_input = Input(shape=(train_xd.shape[1], train_xd.shape[2]), name='Transformer_input')
    x = Dense(64)(transformer_input)  # Project to 64 dimensions
    pos_encoding = PositionalEncoding(position=train_xd.shape[1], d_model=64)(x)
    for _ in range(2):  # Fixed number of layers for simplicity
        x = transformer_block(x, head_size=transformer_head_size, num_heads=transformer_num_heads, 
                             ff_dim=transformer_ff_dim, dropout=transformer_dropout)
    # Use mean pooling instead of last timestep
    transformer_output = Dense(1)(tf.reduce_mean(x, axis=1))

    # NN for time-independent data
    dnn_input = Input(shape=(train_xi.shape[1],), name='NN_input')
    dnn_layer = Dense(nn_neurons, activation='relu')(dnn_input)
    dnn_layer = Dropout(nn_dropout_rate)(dnn_layer)
    dnn_layer = Dense(nn_neurons, activation='relu')(dnn_layer)
    dnn_layer = Dropout(nn_dropout_rate)(dnn_layer)
    dnn_dense = Dense(nn_dense_neurons, activation='linear')(dnn_layer)

    # Combine outputs
    concat = concatenate([transformer_output, dnn_dense], name='Concatenate')
    final_output = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[transformer_input, dnn_input], outputs=final_output, name='Transformer_NN_Model')

    # Compile and train
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit([train_xd, train_xi], train_y, epochs=1000, batch_size=32,  # Adjusted batch size
                        validation_data=([test_xd, test_xi], test_y), shuffle=False, verbose=0, callbacks=[callback])

    tf.keras.backend.clear_session()
    return np.min(history.history['val_loss'])

def rangeCheck(x):
    for i in range(len(x)):
        if x[i] < Lb[i]:
            x[i] = Lb[i]
        if x[i] > Ub[i]:
            x[i] = Ub[i]

nVar = 8  # Number of variables
Ub = np.array([200, 64, 8, 256, 20, 0.5, 0.5, 0.1])  # Adjusted head_size upper bound
Lb = np.array([5, 2, 2, 32, 1, 0.0, 0.0, 0.0001])
maxiter = 50
npop = 10

# OMA Naked Eye Phase
x = np.zeros((npop, nVar))
fit = np.zeros((npop, 1))
x[0, :] = np.random.rand(1, nVar)
for i in range(1, npop):
    x[i, :] = x[i-1, :] * (1 - x[i-1, :])
for i in tqdm(range(npop), desc="Loading..."):
    x[i, :] = Lb + x[i, :] * (Ub - Lb)
    fit[i, :] = objfun(x[i, :])

# OMA Main Loop
A = []
for it in range(maxiter):
    idx = np.argmin(fit)
    bestsol = x[idx, :]
    bestmag = fit[idx, :]
    for i in tqdm(range(npop), desc="Loading..."):
        # Objective Lens Phase
        xnew = bestsol + np.random.rand(nVar) * 1.4 * (x[i, :])
        rangeCheck(xnew)
        fitnew = objfun(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew
        
        # Eyepiece Phase
        j = np.random.randint(0, npop)
        while j == i:
            j = np.random.randint(0, npop)
        if fit[i, :] >= fit[j, :]:
            space = x[j, :] - x[i, :]
        else:
            space = x[i, :] - x[j, :]
        xnew = x[i, :] + np.random.rand(nVar) * 0.55 * space
        rangeCheck(xnew)
        fitnew = objfun(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew
    A.append(bestmag[0])

# Plot OMA convergence
plt.figure()
plt.plot(A)
plt.title('OMA Convergence')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value (Validation Loss)')
plt.savefig('oma_convergence.png')  # Save instead of show

# Print optimal parameters
idx = np.argmin(fit)
bestsol = x[idx, :]
bestmag = fit[idx, :]
print(f"MSE = {bestmag[0]:.6f}")
print("\nOptimal Parameters:")
print(f"NN Number of Neurons           = {int(bestsol[0])}")
print(f"Transformer Head Size          = {int(bestsol[1])}")
print(f"Transformer Number of Heads    = {int(bestsol[2])}")
print(f"Transformer FF Dimension       = {int(bestsol[3])}")
print(f"NN Dense Neurons              = {int(bestsol[4])}")
print(f"Transformer Dropout Rate       = {bestsol[5]:.4f}")
print(f"NN Dropout Rate               = {bestsol[6]:.4f}")
print(f"Learning Rate                 = {bestsol[7]:.4f}")

# Train model with optimal parameters
nn_neurons = int(bestsol[0])
transformer_head_size = int(bestsol[1])
transformer_num_heads = int(bestsol[2])
transformer_ff_dim = int(bestsol[3])
nn_dense_neurons = int(bestsol[4])
transformer_dropout = bestsol[5]
nn_dropout_rate = bestsol[6]
lr = bestsol[7]

# Build and train final model
transformer_input = Input(shape=(train_xd.shape[1], train_xd.shape[2]), name='Transformer_input')
x = Dense(64)(transformer_input)
pos_encoding = PositionalEncoding(position=train_xd.shape[1], d_model=64)(x)
for _ in range(2):
    x = transformer_block(x, head_size=transformer_head_size, num_heads=transformer_num_heads, 
                         ff_dim=transformer_ff_dim, dropout=transformer_dropout)
transformer_output = Dense(1)(tf.reduce_mean(x, axis=1))

dnn_input = Input(shape=(train_xi.shape[1],), name='NN_input')
dnn_layer = Dense(nn_neurons, activation='relu')(dnn_input)
dnn_layer = Dropout(nn_dropout_rate)(dnn_layer)
dnn_layer = Dense(nn_neurons, activation='relu')(dnn_layer)
dnn_layer = Dropout(nn_dropout_rate)(dnn_layer)
dnn_dense = Dense(nn_dense_neurons, activation='linear')(dnn_layer)

concat = concatenate([transformer_output, dnn_dense], name='Concatenate')
final_output = Dense(1, activation='sigmoid')(concat)
final_model = Model(inputs=[transformer_input, dnn_input], outputs=final_output, name='Transformer_NN_Model')

final_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, 
                                               save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
history = final_model.fit([train_xd, train_xi], train_y, epochs=500, batch_size=32, 
                         validation_data=([test_xd, test_xi], test_y), callbacks=[checkpoint], shuffle=False)

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.savefig('training_history.png')  # Save instead of show

# Optimize output weights
final_model.load_weights('best_weights.hdf5')
concat_model = Model(inputs=final_model.input, outputs=final_model.get_layer('Concatenate').output)
concatenated_output = concat_model.predict([train_xd, train_xi])

data = concatenated_output
targets = train_y.values.reshape(-1)

def fitness(weights, data, targets):
    predictions = expit(np.dot(data, weights))
    error = np.mean((targets - predictions) ** 2.0)
    return error

def objfun1(weights):
    return fitness(weights, data, targets)

def rangeCheck1(x):
    for i in range(len(x)):
        if x[i] < Lb1:
            x[i] = Lb1
        if x[i] > Ub1:
            x[i] = Ub1

nVar1 = concatenated_output.shape[1]
Ub1 = 1
Lb1 = -1
maxiter1 = 100
npop1 = 100

# OMA for weight optimization
x = np.zeros((npop1, nVar1))
fit = np.zeros((npop1, 1))
x[0, :] = np.random.rand(1, nVar1)
for i in range(1, npop1):
    x[i, :] = x[i-1, :] * (1 - x[i-1, :])
for i in range(npop1):
    x[i, :] = Lb1 + x[i, :] * (Ub1 - Lb1)
    fit[i, :] = objfun1(x[i, :])

for it in range(maxiter1):
    idx = np.argmin(fit)
    bestsol1 = x[idx, :]
    bestmag1 = fit[idx, :]
    for i in range(npop1):
        xnew = bestsol1 + np.random.rand(nVar1) * 1.4 * (x[i, :])
        rangeCheck1(xnew)
        fitnew = objfun1(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew
        j = np.random.randint(0, npop1)
        while j == i:
            j = np.random.randint(0, npop1)
        if fit[i, :] >= fit[j, :]:
            space = x[j, :] - x[i, :]
        else:
            space = x[i, :] - x[j, :]
        xnew = x[i, :] + np.random.rand(nVar1) * 0.55 * space
        rangeCheck1(xnew)
        fitnew = objfun1(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew

idx = np.argmin(fit)
bestsol1 = x[idx, :]
bestmag1 = fit[idx, :]
print(f"Train_MSE: {bestmag1[0]:.6f}")

# Print optimal weights
for i, weight in enumerate(bestsol1):
    print(f"W{i+1} = {weight:.6f}")
print(f"The sum of the weights: {np.sum(bestsol1):.6f}")

# Predict using optimized model
y_out_concat = concat_model.predict([test_xd, test_xi])
y_pred_OMA = expit(np.dot(y_out_concat, bestsol1))

# Denormalize outputs
max_y = 1
min_y = 0
test_y_act = test_y * (max_y - min_y) + min_y
test_y_act = test_y_act.values.reshape(-1)
y_pred_OMA_act = y_pred_OMA * (max_y - min_y) + min_y
y_pred_OMA_act = y_pred_OMA_act.reshape(-1)
y_pred = final_model.predict([test_xd, test_xi])
y_pred_act = y_pred * (max_y - min_y) + min_y
y_pred_act = y_pred_act.reshape(-1)

# Evaluation metrics for OMA-optimized model
corr_oma, _ = pearsonr(test_y_act, y_pred_OMA_act)
r2_oma = r2_score(test_y_act, y_pred_OMA_act)
mape_oma = mean_absolute_percentage_error(test_y_act, y_pred_OMA_act)
mae_oma = mean_absolute_error(test_y_act, y_pred_OMA_act)  # Use denormalized values
mse_oma = mean_squared_error(test_y_act, y_pred_OMA_act, squared=True)
rmse_oma = mean_squared_error(test_y_act, y_pred_OMA_act, squared=False)

print("\nOMA-Optimized Model Evaluation:")
print(f"R: {corr_oma:.4f}")
print(f"R2: {r2_oma:.4f}")
print(f"MSE: {mse_oma:.4f}")
print(f"RMSE: {rmse_oma:.4f}")
print(f"MAE: {mae_oma:.4f}")
print(f"MAPE: {mape_oma:.4f}")

# Evaluation metrics for default model
corr_adam, _ = pearsonr(test_y_act, y_pred_act)
r2_adam = r2_score(test_y_act, y_pred_act)
mape_adam = mean_absolute_percentage_error(test_y_act, y_pred_act)
mae_adam = mean_absolute_error(test_y_act, y_pred_act)  # Use denormalized values
mse_adam = mean_squared_error(test_y_act, y_pred_act, squared=True)
rmse_adam = mean_squared_error(test_y_act, y_pred_act, squared=False)

print("\nDefault Model Evaluation (ADAM):")
print(f"R: {corr_adam:.4f}")
print(f"R2: {r2_adam:.4f}")
print(f"MSE: {mse_adam:.4f}")
print(f"RMSE: {rmse_adam:.4f}")
print(f"MAE: {mae_adam:.4f}")
print(f"MAPE: {mape_adam:.4f}")

# Save evaluation metrics to text file
with open('evaluation_results.txt', 'w') as f:
    f.write("OMA-Optimized Model Evaluation:\n")
    f.write(f"R: {corr_oma:.4f}\n")
    f.write(f"R2: {r2_oma:.4f}\n")
    f.write(f"MSE: {mse_oma:.4f}\n")
    f.write(f"RMSE: {rmse_oma:.4f}\n")
    f.write(f"MAE: {mae_oma:.4f}\n")
    f.write(f"MAPE: {mape_oma:.4f}\n\n")
    
    f.write("Default Model Evaluation (ADAM):\n")
    f.write(f"R: {corr_adam:.4f}\n")
    f.write(f"R2: {r2_adam:.4f}\n")
    f.write(f"MSE: {mse_adam:.4f}\n")
    f.write(f"RMSE: {rmse_adam:.4f}\n")
    f.write(f"MAE: {mae_adam:.4f}\n")
    f.write(f"MAPE: {mape_adam:.4f}\n")

# Save predictions and actual values to text files
np.savetxt('predictions_OMA.txt', y_pred_OMA_act, fmt='%.6f')
np.savetxt('predictions_ADAM.txt', y_pred_act, fmt='%.6f')
np.savetxt('actual_values.txt', test_y_act, fmt='%.6f')