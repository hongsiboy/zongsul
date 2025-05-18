import logging
import shap
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# 로깅 설정: INFO 레벨 이상의 로그를 파일에 기록
logging.basicConfig(filename='shap_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# SHAP 분석 함수 정의
def model_predict(inputs):
    xd_flat = inputs[:, :15]
    xi = inputs[:, 15:]
    xd_reshaped = xd_flat.reshape(-1, 3, 5)
    concat_output = concat_model.predict([xd_reshaped, xi], verbose=0)
    predictions = expit(np.dot(concat_output, bestsol1))
    return predictions

# 데이터 준비 (사용자의 기존 변수 사용 가정)
train_xd_flat = train_xd.reshape(train_xd.shape[0], -1)
test_xd_flat = test_xd.reshape(test_xd.shape[0], -1)
background_data = np.hstack([train_xd_flat, train_xi])[:100]
test_data = np.hstack([test_xd_flat, test_xi])

# SHAP 분석 시작
logging.info("SHAP 분석 시작")

# SHAP Explainer 생성
explainer = shap.KernelExplainer(model_predict, background_data)
logging.info("SHAP Explainer 생성 완료")

# SHAP 값 계산
shap_values = explainer.shap_values(test_data[:10], nsamples=100)
logging.info("SHAP 값 계산 완료")

# SHAP 시각화
feature_names = [f'xd_{i}' for i in range(15)] + [f'xi_{i}' for i in range(14)]
shap.summary_plot(shap_values, test_data[:10], feature_names=feature_names)
plt.savefig('shap_summary.png')
logging.info("SHAP 시각화 저장 완료")

print("SHAP 분석이 완료되었습니다. 결과는 'shap_summary.png'에 저장되었습니다.")