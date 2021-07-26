# 레모네이드 판매 예측
import tensorflow as tf
import pandas as pd
 
###########################
# 데이터를 준비합니다.
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
레모네이드 = pd.read_csv(파일경로)
레모네이드.head()

# 종속변수, 독립변수
독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)
 
# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1]) # 독립 변수 colume의 갯수(온도)
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')
 
# 모델을 학습시킵니다. 
model.fit(독립, 종속, epochs=10000, verbose=0) # epochs : 학습할 횟수 verbose=0 : 학습 출력표시를 없에준다
model.fit(독립, 종속, epochs=10)
 
# 모델을 이용합니다. 
print(model.predict(독립))
print(model.predict([[10]])) # 온도가 15 일 경우 준비해야하는 레모네이드 양