import numpy as np
from sklearn.linear_model import LinearRegression # untuk memanggil library

# Database
# x = Data, y = Target
x = [[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]]
y = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] # hasil data x dikali 2

regr = LinearRegression().fit(x,y) # untuk memfitting grafik
regr.score(x, y) # untuk menentukan grafik

# Data uji
predict = np.array([[120]]) #nilai yang akan di prediksi

# Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))
