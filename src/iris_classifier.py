from sklearn.datasets import load_iris # We use sklearn to load the Iris dataset, a built in data set with various flower samples from three species. 
from sklearn.model_selection import train_test_split # Imports the function that divides data into: Training data (used to instruct the model) , Testing data (used to assess the model)
from sklearn.tree import DecisionTreeClassifier # Imports the Decision Tree machine learning algorithm. A Decision Tree learns a series of questions (decision rules) to classify data.
from sklearn.metrics import accuracy_score #Imports the function used to calculate the model's accuracy.
from sklearn.metrics import confusion_matrix # imports the confusion matrix function which allows the system to make correct and incorrect predictions for each class                                                
from sklearn.neighbors import KNeighborsClassifier #Imports the k-Nearest Neighbours (k-NN) algorithm, which uses the closest training instances to predict a class.

iris=load_iris() # This loads the Iris dataset into the iris variable.

X = iris.data 
y = iris.target

print(iris.feature_names,iris.target_names) # This yields the flower labels (y) as well as the feature data (X).

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42) # To assess the model's performance, we reserved a subset of the dataset as a test set.

model = DecisionTreeClassifier(random_state=42) # Here we create an instance for the classifier.

model.fit(X_train,y_train) # Because of SciKit-learn's consistent API design, training can be completed for any model in the library. fit(X, y), independent of the kind of model.

y_pred = model.predict(X_test) # once the model has been trained, The species of flowers in our test set can now be predicted using this code line.

print("Predicrtions:" , y_pred[:5])
print("True labels:",y_test[:5]) # These code lines allow us to see how our model is performing, and it is useful to compare a few predictions to the true labels

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy) # The percentage of accurate predictions relative to all predictions is a straightforward metric. By comparing y_pred and y_test and determining the fraction that is equal, we can calculate the accuracy.

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm) # The confusion matrix offers a more thorough analysis of predictions versus genuine labels, and helps understand which classes your model gets right and which classes it confuses, making it more informative than accuracy alone.

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred_knn)) # We compare the performance of a k-NN (k-Nearest Neighbors) model to our original decision tree to demonstrate the iterative process of model improvement.

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# We may notice overfitting in our decision tree if it achieves 100% accuracy on the training set but only 90% on the test set.
# Adjusting hyperparameters like "max_depth" is essential in these situations.
# by keeping the model from picking up noise from the training data, limiting the tree's depth (for example, to three levels) could greatly increase test accuracy while also marginally lowering the model's training accuracy.





# SUMMARY OF RESULTS

# With 100% accuracy on the test set, the decision tree model demonstrated excellent performance on the iris dataset. 
# All of the predictions were accurate, according to the confusion matrix, and the model was able to distinguish between the three species of iris. 
# For comparison, we also evaluated a k-Nearest Neighbours model that attained 100% accuracy.
# We tried restricting the depth of the decision tree to three layers in order to avoid any overfitting. This is significant because models may memorise training material rather than picking up on broad trends.
