# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# Leitura do Arquivo .csv
df=pd.read_csv("/kaggle/input/simple-rainfall-classification-dataset/rainfall.csv")

# Visualizando os dados do arquivo
df.head()

# Visualizando informações gerais do arquivo.
df.info()

# Visualizando tamanho do arquivo
df.shape

# Quantidade de registros faltantes
df.isnull().sum()

# Retirando registros com dados nulos
df = df.dropna()

# Retirando campos desnecessários.
df = df.drop(['date'], axis=1)

# Visualizando limpeza de registros faltantes.
df.isnull().sum()

# Visualizando quantidade agrupada de variável alvo.
df['weather_condition'].value_counts()

# Gráfico de barras para visualização de comparação entre as informações da variável alvo.
custom_palette = {'Rainy': 'lightblue', 'Sunny': 'darkorange'}
sns.countplot(x='weather_condition', data=df, palette=custom_palette)
plt.show()

# Criação dos dataframes de variáveis alvo e variáveis preditoras.
t = df['weather_condition']
p = df.drop('weather_condition', axis = 1)

# Tamanho da variável alvo.
t.shape

# Visualizando a variável alvo.
t.head()

# Tamanho das variáveis preditoras
p.shape

# Visualizando as variáveis preditoras.
p.head()

# Separação dos dataframes de treino e teste para o modelo, considerando 30% para teste e 70% para treino.
from sklearn.model_selection import train_test_split

t_train, t_test, p_train, p_test = train_test_split(t, p, test_size = 0.3)

# Visualizando a separação de 30% do dataframe para teste.
t_test.shape

t_train.shape

# Visualizando a separação de 70% do dataframe para treino.Visualizando a separação de 70% do dataframe para treino.
p_test.shape

p_train.shape

# Chamando a função ExtraTressClassifier (Árvore de decisão) para treinar de acordo com 70% do dataframe com as variáveis preditoras e alvo.
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(p_train, t_train)

# Visualizando a performance do modelo obtido, aplicando as variáveis a serem testadas.
result = (model.score(p_test, t_test))*100
print(f'Performance:  {result}' '%')

# Visualizando os resultados dos testes para 3 registros.
t_test[10:13]

p_test[10:13]

# Validação do modelo de árvore de decisão.
pred = model.predict(p_test[10:13])
print(pred)