import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train_val)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_bias = np.c_[np.ones([X_train_scaled.shape[0], 1]), X_train_scaled]
X_val_scaled_bias = np.c_[np.ones([X_val_scaled.shape[0], 1]), X_val_scaled]
X_test_scaled_bias = np.c_[np.ones([X_test_scaled.shape[0], 1]), X_test_scaled]

def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot

y_train_one_hot = to_one_hot(y_train)
y_val_one_hot = to_one_hot(y_val)
y_test_one_hot = to_one_hot(y_test)

def softmax(logits):
    exps = np.exp(logits)
    exps_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exps_sums

def compute_loss(y_one_hot, probabilities):
    m = len(y_one_hot)
    epsilon = 1e-7
    loss = - (1/m) * np.sum(y_one_hot * np.log(probabilities + epsilon))
    return loss

n_inputs = X_train_scaled_bias.shape[1] 
n_outputs = len(np.unique(y)) 

learning_rate = 0.1
n_epochs = 5001
patience = 50 

np.random.seed(42)
Thetas = np.random.randn(n_inputs, n_outputs)

best_loss = np.inf
epochs_without_improvement = 0
best_thetas = None

m_train = X_train_scaled.shape[0]

print("--- 2개 특성 + 스케일링 모델 훈련 시작 ---")

for epoch in range(n_epochs):
    logits_train = X_train_scaled_bias.dot(Thetas)
    probabilities_train = softmax(logits_train)
    gradients = (1 / m_train) * X_train_scaled_bias.T.dot(probabilities_train - y_train_one_hot)
    Thetas = Thetas - learning_rate * gradients
    
    if epoch % 10 == 0:
        logits_val = X_val_scaled_bias.dot(Thetas)
        probabilities_val = softmax(logits_val)
        val_loss = compute_loss(y_val_one_hot, probabilities_val)
        train_loss = compute_loss(y_train_one_hot, probabilities_train)
        print(f"Epoch {epoch:4d} | 훈련 손실: {train_loss:.4f} | 검증 손실: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_thetas = Thetas.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\n--- 조기 종료! (Epoch: {epoch}) ---")
            print(f"최적 검증 손실: {best_loss:.4f}")
            break

print("훈련이 완료되었습니다.")

def predict(X_with_bias, Thetas):
    logits = X_with_bias.dot(Thetas)
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)

y_pred_test = predict(X_test_scaled_bias, best_thetas)
accuracy = np.mean(y_pred_test == y_test)
print(f"\n최종 모델 테스트 세트 정확도: {accuracy * 100:.2f}%")

plt.figure(figsize=(10, 6))

x0_min, x0_max = 0.8, 7.2
x1_min, x1_max = 0.0, 3.5 

x0, x1 = np.meshgrid(np.linspace(x0_min, x0_max, 500), np.linspace(x1_min, x1_max, 500))

X_mesh_original = np.c_[x0.ravel(), x1.ravel()]

X_mesh_scaled_for_pred = scaler.transform(X_mesh_original)

X_mesh_scaled_bias_for_pred = np.c_[np.ones([X_mesh_scaled_for_pred.shape[0], 1]), X_mesh_scaled_for_pred]

y_mesh_pred = predict(X_mesh_scaled_bias_for_pred, best_thetas)
Z = y_mesh_pred.reshape(x0.shape)

logits_mesh = X_mesh_scaled_bias_for_pred.dot(best_thetas)
probabilities_mesh = softmax(logits_mesh)
P_versicolor = probabilities_mesh[:, 1].reshape(x0.shape)

custom_cmap = ListedColormap(['#FFFFAA', '#AAAAFF', '#AAFFAA']) 
plt.contourf(x0, x1, Z, cmap=custom_cmap)

levels = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90]) 
plt.contour(x0, x1, P_versicolor, levels=levels, colors='maroon', alpha=0.8, linestyles='solid', linewidths=1.5)
plt.clabel(plt.contour(x0, x1, P_versicolor, levels=levels, colors='maroon', alpha=0.8), inline=1, fontsize=10, fmt='%1.2f')

plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="upper left")
plt.title("Softmax Regression Decision Boundaries and Probabilities")
plt.axis([x0_min, x0_max, x1_min, x1_max])
plt.show()