from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix                                              
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris() 

X = iris.data 
y = iris.target

print(iris.feature_names,iris.target_names) 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42) 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test) 

print("Predicrtions:" , y_pred[:5])
print("True labels:",y_test[:5]) 

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy) 

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm) 

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred_knn)) 

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

