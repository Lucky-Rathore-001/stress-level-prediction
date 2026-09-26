import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

df=pd.read_csv("stress_data.csv")

x=df[["Day","Steps_Km","Screen_time_hours","Sleep_hours","Work_hours"]]
y=df[["Stress_level"]]


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

x_train, x_test, y_train, y_test =train_test_split(x_scaled,y,test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(x_train,y_train)


y_pred=model.predict(x_test)

print("stres level=",y_pred)

print("Actual values=",y_test.values)

print("MAE:",mean_absolute_error(y_test,y_pred))


plt.figure(figsize=(6,7))

plt.plot(y_test.values,marker="o",color="red",label="Actual")
plt.plot(y_pred,marker="o",color="blue",label="Model_Prediction")
plt.xlabel("Test Data")
plt.ylabel("Stress Level")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

print("-----USER INPUT-------")

day=float(input("Enter Day:")) 
steps_km=float(input("Enter Steps KM:"))
screen_time_hours=float(input("Enter Screen Time Hour:"))
sleep=float(input("Enter Sleep Hours:"))
work=float(input("Enter Work Hours:"))

user_data=[[day,steps_km,screen_time_hours,sleep,work]]

user_data_scaled=scaler.transform(user_data)

predictionn=model.predict(user_data_scaled)

print("Prediction Stress Level:",predictionn[0])