#importing pandas to be called as pd.
import pandas as pd
#importing tensorflow to be called as tf.
import tensorflow as tf
#importing math plot libary as plt.
import matplotlib.pyplot as plt
#importing test and training splits from sci kit learn
from sklearn.model_selection import train_test_split

#reading csv file for data
diabetes = pd.read_csv('pima-indians-diabetes.csv')

#printing data
print(diabetes.head())

#printing columns
print("\n")
print(diabetes.columns)

#cleaning up data, removing end column data
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min()) / (x.max() - x.min()))

print("\n")
print(diabetes.head())

print("\n")
print(diabetes.columns)

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

print("\n")
print(diabetes['Age'].hist(bins=20))

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])

feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]

print("\n")
print(diabetes.head())

print("\n")
print(diabetes.info())

x_data = diabetes.drop('Class',axis=1)

labels = diabetes['Class']

X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)

model.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

results = model.evaluate(eval_input_func)

print("\next")
print(results)

pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

predictions = model.predict(pred_input_func)

print(list(predictions))

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)

feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,embedded_group_column, age_buckets]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

dnn_model.train(input_fn=input_func,steps=1000)