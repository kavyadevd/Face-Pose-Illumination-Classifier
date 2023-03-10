# **Face, Pose, Illumination Classifier**

<p>
The aim of this project is to implement different classifiers to achieve face recognition given a set of faces and their corresponding labels. You need to split the data into training and
testing sets and use the training data to train your classifiers. The whole pipeline is described in
the following section.
</p>

<p align="center">
  <img alt="Setup1" src="https://user-images.githubusercontent.com/13993518/224439297-9a63f64f-4c97-4986-87b7-ea301478b4a8.png" width="55%">
&nbsp; &nbsp; &nbsp; &nbsp;
</p>


>> This is a midterm project completed for ENEE633/CMSC828C Statistics and Pattern Recognition class University of Maryland College Park

#### Classifiers implemented:
    Bayes’ Classifier
    KNN
    Kernel SVM
    Boosted SVM

#### Dimensionality reduction methods implemented:
    PCA
    MDA

#### Dataset:
data.mat

	200 subjects
	3 faces per subject
	size: 24 x 21

	The file 'data.mat' has a variable ”face” of size (24x21x600). The images corresponding to the
	person labeled n, n = {1, . . . , 200}, can be indexed in Matlab as face(:,:,3*n-2), face(:,:,3*n-1)
	and face(:,:,3*n). The first image is a neutral face, the second image is a face with facial
	expression, and the third image has illumination variations.

pose.mat

	68 subjects
	13 images per subject (13 different poses)
	size: 48 x 40

	The file 'pose.mat' has a variable "pose" of size 48x40x13x68. 
	pose(:,:,i,j) gives i^th image of j^th subject.


illumination.mat

	68 subjects
	21 images per subject (21 different illuminations)
	size: 48x40

	The file 'illumination.mat' has a variable "illum" of size 1920x21x68.
	reshape(illum(:, i,j), 48, 40) gives i^th image of j^th subject.

# Steps to run:
### Wrapper code that calls the methods to classify is present in the Jupyter notebooks
Load the notebook in any compatible IDE and click run all.</br>
**All notebooks eveluate the classification first without dimensionality reduction, then with various variations of PCA followed by various variations of MDA**
### Part 1:
1. **Bayes classifier: Bayes.ipynb** -> runs illumination testing data against neutral and facial expression trainingdata for data.mat and prints accuracy at the end of each cell after classification. Similiar patter is followed forother data sets.</br>
All 3 datasets are evaluated in this notebook

2. **k-Nearest Neighbors : KNN.ipynb** -> ( _Same as above_ ) runs illumination testing data against neutral and facialexpression training data for data.mat and prints accuracy at the end of each cell after classification. Similiarpatter is followed for other data sets.</br>
All 3 datasets are evaluated in this notebook

### Part 2:
1. **Bayes classifier: Bayes_Q2.ipynb** -> Evaluates the data.mat data to classify neutral vs facial expression input with PCA, MDA and without it
2. **k-Nearest Neighbors: KNN_Q2.ipynb** -> ( _Same as above_ ) Evaluates the data.mat data to classify neutral vs facial expression input with PCA, MDA and without it
3. **Kernel SVM: SVM.ipynb** -> Here the classification is done with three different kernels: rbf, polynomial and linear.
3. **Boosted Kernel SVM: SVM.ipynb** -> Here the classification is done with PCA and using a linear kernel.


### Helper classes:
The helper classes and methods are present in the .py files:

Read and load train/test data:

    get_train_test_data_2.py
    get_train_test_data1.py

PCA, MDA:

    pca.py
    mda.py

Classifiers:

    bayes_classifier.py
    knn.py
    svm.py

### Some of the Graph outputs are present in the result folder

