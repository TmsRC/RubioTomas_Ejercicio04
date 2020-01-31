import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
from itertools import combinations

data = pd.read_csv('Cars93.csv')
Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)

rs = []
coefs = []
for i in range(1,12):
    rs_i = []
    indices = combinations(range(0,11),i)
    for c in list(indices):
        X_sub = X_train[:,list(c)]
        regresion = sklearn.linear_model.LinearRegression()
        regresion.fit(X_sub, Y_train)
        rs_i.append(regresion.score(X_test[:,list(c)],Y_test))
    rs.append(rs_i)
rs = np.array(rs)

plt.figure()
maximos = []
for i in range(0,11):
    plt.scatter((i+1)*np.ones(len(rs[i])),rs[i])
    maximos.append(np.max(rs[i]))
plt.plot(range(1,12),maximos,c='black')
plt.savefig('R_2.png')