Iris Flower Dataset
Iris flower data set used for multi-class classification.

About Dataset Context
The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper 
The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation 
of Iris flowers of three related species. 
The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). 
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. 
The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).
This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines. 
This dataset is widely used for introductory classification tasks.

Implementation Details:

Classify iris flowers using machine learning
Classification of iris flowers is perhaps the best-known example of machine learning.
The aim is to classify iris flowers among three species (Setosa, Versicolor, or Virginica) from the sepals' and petals' length and width measurements.
Here, we design a model that makes proper classifications for new flowers.
 It involves data preprocessing, model training, and evaluation, showcasing a fundamental classification task.
Find the accuracy of the model and view the confusion matrix. 
The accuracy score tells us how accurately the model we build will predict and the confusion matrix has a matrix with Actual values and predicted values. 
For that, import accuracy_score and confusion_matrix from the sci-kit learn metric library.

Classification problem: our goal is to predict the flow 'Species' with given features


<h3>Steps to be followed:</h3>

<b>Understanding the Problem:</b> 
Familiarize yourself with the Iris flower dataset and the problem at hand. 
Understand the features of the dataset and the goal of classifying iris flowers into different species.

<b>Literature Review:</b>
Conduct a literature review to understand existing approaches and models for multi-model classification. 
This will help you identify best practices and state-of-the-art techniques.

<b>Data Exploration:</b>
Explore the Iris dataset to gain insights into its structure, features, and any patterns that may exist. 
Visualize the data using plots and graphs to better understand the relationships between different features.

<b>Preprocessing:</b>
Clean and preprocess the data as needed. 
This may involve handling missing values, encoding categorical variables, and scaling numerical features. 
Ensure that the data is ready for model training.

<b>Model Selection:</b>
Research and experiment with different machine learning models suitable for multi-class classification. 
Common models for this type of problem include decision trees, random forests, support vector machines, and neural networks.

<b>Model Training:</b>
Implement and train the selected models on the Iris dataset. 
Fine-tune hyperparameters to achieve the best performance. 
Use techniques like cross-validation to assess the model's generalization capabilities.

<b>Evaluation:</b>
Evaluate the performance of the trained models using appropriate metrics such as accuracy, precision, recall, and F1 score. 
Compare the results to determine the most effective model for the Iris flower classification task.


<b>Conclusion: </b>

We took Iris Flowers dataset from kaggle. Studied statistical description of the dataset with pandas. 
matplotlib and seaborn are used for plotting visualizations of various features and performed a logistic regression algorithm with sklearn to design a model 
that classifies flowers into their species with accuracy score of 97.37%.

