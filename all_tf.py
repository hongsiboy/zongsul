import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, LayerNormalization, MultiHeadAttention, Layer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr
from scipy.special import expit
import matplotlib.pyplot as plt
from tqdm import tqdm

# 위치 인코딩 레이어 정의
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

# Transformer 블록 정의
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

# 데이터 로드 및 전처리
raw_csv_data = pd.read_csv(r"ESTC Denorm.csv")
df = raw_csv_data.copy()
x = df.values
df = pd.DataFrame(x)

split = int(df.shape[0] * 0.8)
train_dataset = df.loc[:split, :]
test_dataset = df.loc[split:, :]

train_xi = train_dataset.loc[:, 1:14]  # 비시계열 데이터
test_xi = test_dataset.loc[:, 1:14]
train_xg = train_dataset.loc[:, 15:]   # 시계열 데이터
test_xg = test_dataset.loc[:, 15:]

train_xd = train_xg.values.reshape(train_xg.shape[0], 3, 5)  # (샘플, 타임스텝, 특징)
test_xd = test_xg.values.reshape(test_xg.shape[0], 3, 5)

train_y = train_dataset.iloc[:, 0]
test_y = test_dataset.iloc[:, 0]

# 두 Transformer로 구성된 모델 정의
def build_transformer_transformer_model(train_xd_shape, train_xi_shape, params):
    K = 10  # 비시계열 데이터의 토큰 수
    d_model = 64

    # 시계열 데이터 Transformer 브랜치
    transformer_input_td = Input(shape=train_xd_shape[1:], name='Transformer_input_td')
    x_td = Dense(d_model)(transformer_input_td)  # d_model로 투영
    x_td = PositionalEncoding(position=train_xd_shape[1], d_model=d_model)(x_td)
    for _ in range(2):
        x_td = transformer_block(x_td, params['head_size'], params['num_heads'], params['ff_dim'], params['dropout'])
    output_td = tf.reduce_mean(x_td, axis=1)  # (batch, d_model)

    # 비시계열 데이터 Transformer 브랜치
    transformer_input_ti = Input(shape=(train_xi_shape[1],), name='Transformer_input_ti')
    x_ti = Dense(K * d_model)(transformer_input_ti)  # (batch, K * d_model)
    x_ti = tf.reshape(x_ti, (-1, K, d_model))  # (batch, K, d_model)
    # 위치 인코딩 없이 처리
    for _ in range(2):
        x_ti = transformer_block(x_ti, params['head_size'], params['num_heads'], params['ff_dim'], params['dropout'])
    output_ti = tf.reduce_mean(x_ti, axis=1)  # (batch, d_model)

    # 출력 결합 및 최종 예측
    concat = concatenate([output_td, output_ti], name='Concatenate')  # (batch, 2 * d_model)
    final_output = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[transformer_input_td, transformer_input_ti], outputs=final_output)
    return model

# OMA를 사용한 모델 구조 최적화
def objfun(x):
    head_size = int(x[0])
    num_heads = int(x[1])
    ff_dim = int(x[2])
    dropout = x[3]
    lr = x[4]

    # head_size가 d_model(64)의 약수가 되도록 조정
    possible_head_sizes = [2, 4, 8, 16, 32, 64]
    head_size = min(possible_head_sizes, key=lambda k: abs(k - head_size))

    # 모델 빌드 및 컴파일
    model = build_transformer_transformer_model(train_xd.shape, train_xi.shape, 
                                                {'head_size': head_size, 'num_heads': num_heads, 
                                                 'ff_dim': ff_dim, 'dropout': dropout})
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit([train_xd, train_xi], train_y, epochs=1000, batch_size=32,
                        validation_data=([test_xd, test_xi], test_y), shuffle=False, verbose=0, callbacks=[callback])

    tf.keras.backend.clear_session()
    return np.min(history.history['val_loss'])

def rangeCheck(x, Lb, Ub):
    for i in range(len(x)):
        if x[i] < Lb[i]:
            x[i] = Lb[i]
        if x[i] > Ub[i]:
            x[i] = Ub[i]

nVar = 5  # 변수 수: head_size, num_heads, ff_dim, dropout, lr
Ub = np.array([64, 8, 256, 0.5, 0.1])  # 상한
Lb = np.array([2, 2, 32, 0.0, 0.0001])  # 하한
maxiter = 50
npop = 10

# OMA Naked Eye 단계
x = np.zeros((npop, nVar))
fit = np.zeros((npop, 1))
x[0, :] = np.random.rand(1, nVar)
for i in range(1, npop):
    x[i, :] = x[i-1, :] * (1 - x[i-1, :])
for i in tqdm(range(npop), desc="Loading..."):
    x[i, :] = Lb + x[i, :] * (Ub - Lb)
    fit[i, :] = objfun(x[i, :])

# OMA 메인 루프
A = []
for it in range(maxiter):
    idx = np.argmin(fit)
    bestsol = x[idx, :]
    bestmag = fit[idx, :]
    for i in tqdm(range(npop), desc="Loading..."):
        # Objective Lens 단계
        xnew = bestsol + np.random.rand(nVar) * 1.4 * (x[i, :])
        rangeCheck(xnew, Lb, Ub)
        fitnew = objfun(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew
        
        # Eyepiece 단계
        j = np.random.randint(0, npop)
        while j == i:
            j = np.random.randint(0, npop)
        if fit[i, :] >= fit[j, :]:
            space = x[j, :] - x[i, :]
        else:
            space = x[i, :] - x[j, :]
        xnew = x[i, :] + np.random.rand(nVar) * 0.55 * space
        rangeCheck(xnew, Lb, Ub)
        fitnew = objfun(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew
    A.append(bestmag[0])

# OMA 수렴 플롯
plt.figure()
plt.plot(A)
plt.title('OMA Convergence')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value (Validation Loss)')
plt.savefig('oma_convergence.png')

# 최적 파라미터 출력
idx = np.argmin(fit)
bestsol = x[idx, :]
bestmag = fit[idx, :]
print(f"MSE = {bestmag[0]:.6f}")
print("\nOptimal Parameters:")
print(f"Head Size          = {int(bestsol[0])}")
print(f"Number of Heads    = {int(bestsol[1])}")
print(f"FF Dimension       = {int(bestsol[2])}")
print(f"Dropout Rate       = {bestsol[3]:.4f}")
print(f"Learning Rate      = {bestsol[4]:.4f}")

# 최적 파라미터로 모델 학습
head_size = int(bestsol[0])
num_heads = int(bestsol[1])
ff_dim = int(bestsol[2])
dropout = bestsol[3]
lr = bestsol[4]

model = build_transformer_transformer_model(train_xd.shape, train_xi.shape, 
                                            {'head_size': head_size, 'num_heads': num_heads, 
                                             'ff_dim': ff_dim, 'dropout': dropout})
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, 
                                               save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
history = model.fit([train_xd, train_xi], train_y, epochs=500, batch_size=32, 
                    validation_data=([test_xd, test_xi], test_y), callbacks=[checkpoint], shuffle=False)

# 학습 히스토리 플롯
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.savefig('training_history.png')

# 출력 가중치 최적화
model.load_weights('best_weights.hdf5')
concat_model = Model(inputs=model.input, outputs=model.get_layer('Concatenate').output)
concatenated_output = concat_model.predict([train_xd, train_xi])

data = concatenated_output
targets = train_y.values.reshape(-1)

def fitness(weights, data, targets):
    predictions = expit(np.dot(data, weights))
    error = np.mean((targets - predictions) ** 2.0)
    return error

def objfun1(weights):
    return fitness(weights, data, targets)

def rangeCheck1(x, Lb1, Ub1):
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

# OMA로 가중치 최적화
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
        rangeCheck1(xnew, Lb1, Ub1)
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
        rangeCheck1(xnew, Lb1, Ub1)
        fitnew = objfun1(xnew)
        if fitnew < fit[i, :]:
            fit[i, :] = fitnew
            x[i, :] = xnew

idx = np.argmin(fit)
bestsol1 = x[idx, :]
bestmag1 = fit[idx, :]
print(f"Train_MSE: {bestmag1[0]:.6f}")

# 최적 가중치 출력
for i, weight in enumerate(bestsol1):
    print(f"W{i+1} = {weight:.6f}")
print(f"The sum of the weights: {np.sum(bestsol1):.6f}")

# 예측 수행
y_out_concat = concat_model.predict([test_xd, test_xi])
y_pred_OMA = expit(np.dot(y_out_concat, bestsol1))

# 출력 역정규화
max_y = 1
min_y = 0
test_y_act = test_y * (max_y - min_y) + min_y
test_y_act = test_y_act.values.reshape(-1)
y_pred_OMA_act = y_pred_OMA * (max_y - min_y) + min_y
y_pred_OMA_act = y_pred_OMA_act.reshape(-1)
y_pred = model.predict([test_xd, test_xi])
y_pred_act = y_pred * (max_y - min_y) + min_y
y_pred_act = y_pred_act.reshape(-1)

# OMA 최적화 모델 평가
corr_oma, _ = pearsonr(test_y_act, y_pred_OMA_act)
r2_oma = r2_score(test_y_act, y_pred_OMA_act)
mape_oma = mean_absolute_percentage_error(test_y_act, y_pred_OMA_act)
mae_oma = mean_absolute_error(test_y_act, y_pred_OMA_act)
mse_oma = mean_squared_error(test_y_act, y_pred_OMA_act, squared=True)
rmse_oma = mean_squared_error(test_y_act, y_pred_OMA_act, squared=False)

print("\nOMA-Optimized Model Evaluation:")
print(f"R: {corr_oma:.4f}")
print(f"R2: {r2_oma:.4f}")
print(f"MSE: {mse_oma:.4f}")
print(f"RMSE: {rmse_oma:.4f}")
print(f"MAE: {mae_oma:.4f}")
print(f"MAPE: {mape_oma:.4f}")

# 기본 모델 평가 (ADAM)
corr_adam, _ = pearsonr(test_y_act, y_pred_act)
r2_adam = r2_score(test_y_act, y_pred_act)
mape_adam = mean_absolute_percentage_error(test_y_act, y_pred_act)
mae_adam = mean_absolute_error(test_y_act, y_pred_act)
mse_adam = mean_squared_error(test_y_act, y_pred_act, squared=True)
rmse_adam = mean_squared_error(test_y_act, y_pred_act, squared=False)

print("\nDefault Model Evaluation (ADAM):")
print(f"R: {corr_adam:.4f}")
print(f"R2: {r2_adam:.4f}")
print(f"MSE: {mse_adam:.4f}")
print(f"RMSE: {rmse_adam:.4f}")
print(f"MAE: {mae_adam:.4f}")
print(f"MAPE: {mape_adam:.4f}")

# 평가 결과 텍스트 파일로 저장
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

# 예측값 및 실제값 텍스트 파일로 저장
np.savetxt('predictions_OMA.txt', y_pred_OMA_act, fmt='%.6f')
np.savetxt('predictions_ADAM.txt', y_pred_act, fmt='%.6f')
np.savetxt('actual_values.txt', test_y_act, fmt='%.6f')
