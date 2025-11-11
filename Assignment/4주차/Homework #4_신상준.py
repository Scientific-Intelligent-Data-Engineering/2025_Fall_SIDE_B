import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def run_mnist_knn_full():
    print("MNIST 전체 데이터셋 로드 중... (시간이 걸릴 수 있습니다)")
    
    # 1. 데이터 로드 (전체 70,000개)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X, y = mnist["data"], mnist["target"].astype(int)

    # 2. 데이터 분할 (전체 데이터 사용)
    # MNIST 표준 분할: 훈련 60,000개 / 테스트 10,000개
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"데이터 로드 완료. 훈련 세트: {len(y_train)}개, 테스트 세트: {len(y_test)}개")

    # 3. 데이터 스케일링
    # KNN은 거리를 계산하므로 피처 스케일링이 매우 중요합니다.
    print("데이터 스케일링 중...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # 테스트 데이터는 fit하지 않음

    # 4. 모델 생성 및 훈련
    print("KNeighborsClassifier 모델 생성 (k=5)")
    # n_jobs=-1 : 모든 CPU 코어를 사용하여 이웃 검색 속도를 높임 (필수)
    # weights='distance' : 가까운 이웃에 더 큰 가중치를 줌
    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)

    print("모델 훈련 시작... (KNN은 훈련이 빠릅니다)")
    start_time = time.time()
    knn_clf.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"모델 훈련 완료. (소요 시간: {end_time - start_time:.2f}초)")

    # 5. 모델 평가 (이 단계가 매우 오래 걸립니다!)
    print("\n모델 평가 시작... (시간이 매우 오래 걸립니다. 기다려주세요...)")
    start_time = time.time()
    y_pred = knn_clf.predict(X_test_scaled)
    end_time = time.time()
    print(f"모델 예측 완료. (소요 시간: {end_time - start_time:.2f}초)") # 여기 소요 시간을 확인하세요

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- 최종 정확도 ---")
    print(f"테스트 세트 정확도: {accuracy * 100:.2f}%")

    print("\n--- 분류 리포트 ---")
    print(classification_report(y_test, y_pred))

    # 6. 샘플 예측 및 시각화
    print("\n--- 샘플 예측 시각화 ---")
    sample_index = 100  # 100번째 테스트 이미지
    some_digit = X_test[sample_index]
    some_digit_image = some_digit.reshape(28, 28)
    some_digit_scaled = X_test_scaled[sample_index].reshape(1, -1)
    
    prediction = knn_clf.predict(some_digit_scaled)

    print(f"모델 예측: {prediction[0]}")
    print(f"실제 레이블: {y_test[sample_index]}")

    plt.imshow(some_digit_image, cmap="binary")
    plt.title(f"Predicted: {prediction[0]}, Actual: {y_test[sample_index]}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_mnist_knn_full()