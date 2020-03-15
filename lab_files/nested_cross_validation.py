from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
#import k-nn classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
#view a description of the dataset (uncomment next line to do so)
#print(iris.DESCR)
#Set X equal to features, Y equal to the targets
X=iris.data
y=iris.target
mySeed=1234567
#initialize random seed generator
np.random.seed(mySeed)
#we add some random noise to our data to make the task more challenging
X=X+np.random.normal(0,0.5,X.shape)
#----------------------------------------------------------------------------#
# nested cross validation function
# X - data / features
# y - outputs
# foldK - number of folds
# nns - list of number of neighbours parameter for validation
# dists - list of distances for validation
# mySeed - random seed
# returns: accuracy over 5 folds (list)

def myNestedCrossVal(X,y,foldK,nns,dists,mySeed):
    np.random.seed(mySeed)
    accuracy_fold=[]

    #TASK: use the function np.random.permutation to generate a list of shuffled indices from in the range (0,number of data)
    #(you did this already in a task above)
    L=list(range(X.shape[0]))
    indices = np.random.permutation(L)
    #TASK: use the function array_split to split the indices to foldK different bins (here, 5)
    #uncomment line below
    bins=np.array_split(indices, foldK)
    #print(bins)
    #no need to worry about this, just checking that everything is OK
    assert(foldK==len(bins))
    #loop through folds
    for i in range(0,foldK):
        foldTrain=[] # list to save current indices for training
        foldTest=[]  # list to save current indices for testing
        foldVal=[]    # list to save current indices for validation
        #loop through all bins, take bin i for testing, the next bin for validation, and the rest for training
        valBin = (i+1)%foldK
        for j in range(0,len(bins)):
            if (i == j):
                foldTest = np.concatenate((foldTest, bins[i]), axis=None)
            elif (j == valBin):
                foldVal = np.concatenate((foldVal, bins[valBin]), axis=None)
            else:
                foldTrain = np.concatenate((foldTrain, bins[j]), axis=None)

        #print('** Train', len(foldTrain), foldTrain)
        #print('** Val', len(foldVal), foldVal)
        #print('** Test', len(foldTest), foldTest)
        #no need to worry about this, just checking that everything is OK
        assert not np.intersect1d(foldTest,foldVal)
        assert not np.intersect1d(foldTrain,foldTest)
        assert not np.intersect1d(foldTrain,foldVal)

        bestDistance='' #save the best distance metric here
        bestNN=-1 #save the best number of neighbours here
        bestAccuracy=-10 #save the best attained accuracy here (in terms of validation)


        # loop through all parameters (one for loop for distances, one for loop for nn)
        # train the classifier on current number of neighbours/distance
        # obtain results on validation set
        # save parameters if results are the best we had
        for d in dists:
            for nn in nns:
                #split to train and test
                X_train, X_test, y_train, y_test = train_test_split(X[foldTrain], y[foldTrain], test_size=0.2)
                #define knn classifier, with 5 neighbors and use the euclidian distance
                knn=KNeighborsClassifier(n_neighbors=10, metric='euclidean')
                knn.fit(X_train,y_train)
                y_pred=knn.predict(X_test)

        #print('** End of val for this fold, best NN', bestNN, 'best Dist', bestDistance)


        #evaluate on test data:
        #extend your training set by including the validation set
        #train k-NN classifier on new training set and test on test set
        #get performance on fold, save result in accuracy_fold

        #print('==== Final Cross-val on test on this fold with NN', bestNN, 'dist', bestDistance, ' accuracy ',accuracy_score(y[foldTest],y_pred))

    return accuracy_fold;

#call your nested crossvalidation function:

accuracy_fold=myNestedCrossVal(X,y,5,list(range(1,11)),['euclidean','manhattan'],mySeed)
print(accuracy_fold)
