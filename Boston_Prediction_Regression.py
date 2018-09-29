import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Ignoring warnings
import warnings
warnings.filterwarnings("ignore")

									# Part - 1 Algorithms

# Data-Preprocessing
data = pd.read_csv('./Datasets/housing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Spliting Data 
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42)

#Feature Scaling the data
from sklearn.preprocessing import StandardScaler
fs_X = StandardScaler()
fs_y = StandardScaler()
X_train = fs_X.fit_transform(X_train1)
X_test = fs_X.transform(X_test1)
y_train = fs_y.fit_transform(np.array(y_train1).reshape(-1, 1))
y_test = fs_y.transform(np.array(y_test1).reshape(-1, 1))

# Linear Regression
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

predictions_lin = lm.predict(X_test)

# Polynomial Regression using degree 3
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, y_train)
lm2 = LinearRegression()
lm2.fit(X_poly, y_train)

predictions_poly = lm2.predict(poly.fit_transform(X_test))

#Suppor Vector Regression
from sklearn.svm import SVR
lm4 = SVR(kernel='rbf')
lm4.fit(X_train, y_train)
 
predictions_svr = lm4.predict(X_test)
predictions_svr= np.reshape(predictions_svr, (147, 1))

# Decision Regression 
from sklearn.tree import DecisionTreeRegressor
lm5 = DecisionTreeRegressor()
lm5.fit(X_train, y_train)

predictions_dr = lm5.predict(X_test)
predictions_dr= np.reshape(predictions_dr, (147, 1))

#Random Forest
from sklearn.ensemble import RandomForestRegressor
lm6 = RandomForestRegressor(n_estimators=1000)
lm6.fit(X_train, y_train)

predictions_rfr = lm6.predict(X_test)
predictions_rfr = np.reshape(predictions_rfr, (147, 1))

# Calculating the Result in terms of errors
from sklearn import metrics
result = list()
result.append(metrics.mean_squared_error(y_test, predictions_lin))
result.append(metrics.mean_squared_error(y_test, predictions_poly))
result.append(metrics.mean_squared_error(y_test, predictions_svr))
result.append(metrics.mean_squared_error(y_test, predictions_dr))
result.append(metrics.mean_squared_error(y_test, predictions_rfr))	
result = np.array(result)


									# Part 2 Visualizations
import os 														# for clearing display
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def display(r1, r2, st):
	fig1 = plt.figure()
	fig1 = plt.scatter(r1, r2)
	plt.title(st)
	plt.xlabel(namestr(r1, globals())[0])
	plt.ylabel(namestr(r2, globals())[0])
	fig3 = plt.figure()
	fig3 = sns.distplot(r2-r1)
	plt.title(st)
	plt.ylabel("Density")
	plt.xlabel("Values")
	plt.show()

while True:	
	os.system('cls') 							
	print("Hi there! Here is the predictions of different Regression Algorithms of Boston Housing\n\n")					
	print("Select a choice\n1. Visualization of Algorithms efficiency\n2. View some testcases\n3. Visualization of Final Result\n4. Exit")
	choice = int(input())
	if choice == 1:
		choice_algo = 0
		while choice_algo!=7:
			os.system('cls')
			print("Select an algo to visualize:\n1.Linear Regression\n2.Polynomial Regression\n3.Support Vector Regression\n4.Decision Regression\n5.Random Forest Regression\n6.All Algorithms\n7.Previous Menu\n") 
			choice_algo = int(input())
			
			if choice_algo==1:
				# Visualization of LR
				display(predictions_lin, y_test, "Linear Regression")
			
			elif choice_algo==2:
				# Visualization of PR - 3
				display(predictions_poly, y_test, "Polynomial Regression")
			
			elif choice_algo==3:
				# Visualization of SVR
				display(predictions_svr, y_test, "Support Vector Regression")
			
			elif choice_algo==4:
				# Visualization of Decision Regresssion
				display(predictions_dr, y_test, "Decision Tree Regression")
			
			elif choice_algo==5:
				# Visualization of Random Forest Regression
				display(predictions_rfr, y_test, "Random Forest Regression")
			
			elif choice_algo==6:
				# All Algos
				display(predictions_lin, y_test, "Linear Regression")
				display(predictions_poly, y_test, "Polynomial Regression")
				display(predictions_svr, y_test, "Support Vector Regression")
				display(predictions_dr, y_test, "Decision Tree Regression")
				display(predictions_rfr, y_test, "Random Forest Regression")

			elif choice_algo==7:
				# Previous Menu
				continue

			else:
				# Invalid input
				print("Please enter a valid option", choice_algo, "is not valid. Press Enter to continue")
				temp = input()

	elif choice==3:
		# Visualizing the Result in terms of errors
		fig13 = plt.figure()
		objects = ('LR', 'Poly', 'SVR', 'DR', 'RFR')
		y_pos = np.arange(len(objects))
		fig13 = plt.bar(y_pos, result)
		plt.title("The Final Result")
		plt.xticks(y_pos, objects)
		fig_txt = 'Least ErrorRate in RFR, It is the best for the dataset'
		plt.text(0.75, 0.23, fig_txt)
		plt.show()

	elif choice == 2:
		# Testcases
		os.system('cls')
		print("\t\t\t\t\t********Testcases********\n")
		print("The following results show the prediction of each algorithm and the actual value of a predefined testcase.\n\n")
		print("S.No\tLinear Regression\tPolynomial Regression\tSV Regression\tDT Regression\tRF Regression\tActual Value")
		print("1\t",round(fs_y.inverse_transform(lm.predict(fs_X.transform(np.array([[5.926, 18.13, 20.2]]))))[0][0], 1),"\t\t", round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[5.926, 18.13,20.2]]))))[0], 0), "\t\t",round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[5.926, 18.13,20.2]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm5.predict(fs_X.transform(np.array([[5.926, 18.13,20.2]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm6.predict(fs_X.transform(np.array([[5.926, 18.13,20.2]]))))[0], 0),"\t",round(y_test1.iloc[0], 2), "\n", end='')
		print("1\t",round(fs_y.inverse_transform(lm.predict(fs_X.transform(np.array([[6.389, 9.62, 18.5]]))))[0][0], 1),"\t\t", round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[6.389, 9.62, 18.5]]))))[0], 0), "\t\t",round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[6.389, 9.62, 18.5]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm5.predict(fs_X.transform(np.array([[6.389, 9.62, 18.5]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm6.predict(fs_X.transform(np.array([[6.389, 9.62, 18.5]]))))[0], 0),"\t",round(y_test1.iloc[1], 2), "\n", end='')
		print("1\t",round(fs_y.inverse_transform(lm.predict(fs_X.transform(np.array([[6.655, 17.73, 20.2]]))))[0][0], 1),"\t\t", round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[6.655, 17.73, 20.2]]))))[0], 0), "\t\t",round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[6.655, 17.73, 20.2]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm5.predict(fs_X.transform(np.array([[6.655, 17.73, 20.2]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm6.predict(fs_X.transform(np.array([[6.655, 17.73, 20.2]]))))[0], 0),"\t",round(y_test1.iloc[2], 2), "\n", end='')
		print("1\t",round(fs_y.inverse_transform(lm.predict(fs_X.transform(np.array([[5.414, 23.97, 20.1]]))))[0][0], 1),"\t\t", round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[5.414, 23.97, 20.1]]))))[0], 0), "\t\t",round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[5.414, 23.97, 20.1]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm5.predict(fs_X.transform(np.array([[5.414, 23.97, 20.1]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm6.predict(fs_X.transform(np.array([[5.414, 23.97, 20.1]]))))[0], 0),"\t",round(y_test1.iloc[3], 2), "\n", end='')
		print("1\t",round(fs_y.inverse_transform(lm.predict(fs_X.transform(np.array([[6.459, 23.98, 20.2]]))))[0][0], 1),"\t\t", round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[6.459, 23.98, 20.2]]))))[0], 0), "\t\t",round(fs_y.inverse_transform(lm4.predict(fs_X.transform(np.array([[6.459, 23.98, 20.2]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm5.predict(fs_X.transform(np.array([[6.459, 23.98, 20.2]]))))[0], 0), "\t", round(fs_y.inverse_transform(lm6.predict(fs_X.transform(np.array([[6.459, 23.98, 20.2]]))))[0], 0),"\t",round(y_test1.iloc[4], 2), "\n", end='')
		print("\n\nPress Enter to go back to the previous menu")
		temp = input()	

	elif choice == 4:
		exit()

	else:
		# Invalid input
		print("Please enter a valid option.", choice, "is not valid. Press Enter to continue")
		temp = input()