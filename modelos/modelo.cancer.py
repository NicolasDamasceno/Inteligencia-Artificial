import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Bloqueia avisos de log do TensorFlow
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

path = '../data/'

previsores = pd.read_csv(f'{path}015_entradas-breast.csv')
classe = pd.read_csv(f'{path}015_saidas-breast.csv')

# Testes
previsores_treino_temp, previsores_teste, classe_treino_temp, classe_teste = train_test_split(
    previsores, classe, test_size=0.20, random_state=42
)

# Validação
previsores_treinamento, previsores_val, classe_treinamento, classe_val = train_test_split(
    previsores_treino_temp, classe_treino_temp, test_size=0.20, random_state=42
)

# Normalização
scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_val = scaler.transform(previsores_val)
previsores_teste = scaler.transform(previsores_teste)

# Construção da Rede Neural
classificador = Sequential([
    Dense(units=64, activation='relu', kernel_initializer='random_uniform', input_dim=30),
    Dropout(0.2),
    Dense(units=32, activation='relu', kernel_initializer='random_uniform'),
    Dropout(0.2),
    Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
    Dense(units=1, activation='sigmoid')
])

otimizador = keras.optimizers.Adam(learning_rate=0.0001)
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

# Configurando o Early Stopping
# monitor: o que observar | patience: quantas épocas esperar sem melhora | restore_best_weights: voltar para o melhor momento
callback_parada = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Construindo Histórico para gerar Gráficos
historico = classificador.fit(
    previsores_treinamento, classe_treinamento,
    batch_size=10,
    epochs=500,
    validation_data=(previsores_val, classe_val), # Validação ativa!
    callbacks=[callback_parada],
    verbose=1
)

# Gráficos de Desempenho
plt.figure(figsize=(12, 5))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(historico.history['binary_accuracy'], label='Treino')
plt.plot(historico.history['val_binary_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico de Perda (Loss)
plt.subplot(1, 2, 2)
plt.plot(historico.history['loss'], label='Treino')
plt.plot(historico.history['val_loss'], label='Validação')
plt.title('Perda por Época')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()

previsores = classificador.predict(previsores_teste)
previsoes = (previsores > 0.9)

# Ver Previsoes
# print(previsores)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print(precisao)
print(matriz)

resultado = classificador.evaluate(previsores_teste, classe_teste)

