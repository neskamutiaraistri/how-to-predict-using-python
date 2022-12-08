from sklearn.preprocessing import PolynomialFeatures # untuk memanggil library
from sklearn import linear_model # untuk memanggil library
import numpy as np 


#database 
#x = data, y = Target 
x = [[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]]
y = [4, 16, 36, 64, 100, 144, 196, 256, 324, 400] # hasil pangkat dari nilai x

#data uji 
predict = np.array([[23]]) # nilai yang akan di prediksi 
poly = PolynomialFeatures(degree=2) # nilai ordo yang digunakan
x_=poly.fit_transform(x) # untuk memfitting prediksi sumbu
predict_ = poly.fit_transform(predict) #untuk memfitting jenis regresi
regr = linear_model.LinearRegression() # untuk meregresi
regr.fit(x_,y) # untuk menentukan grafik

#menampilkan data prediksi 
print ("Prediksi")
print ("input = ", predict)
print ("Output =", regr.predict(predict_))
