#
# Leakage : comparing the models using their test set performance is textbook leakage,
#           where we inform our modelling with some information from the test set.
#           Therefore the test set is no longer an honest reporter of generalized performance.
#           Instead, we should use some sort of cross-validation scheme to compare the models
#
# no scaling : both KNN and logistic regression will be influenced by features scales, so if 
#              are not scaled we are introducting some inherent bias toward the large features 
#
# no hyper-parameter optimization: here we rely on the default values of the hyper-parameters 
#                                  but there is no reason to beliave they are appropriate for this data
# 
# One Hot Encoding : use the drop_first option or you have colinearity / redundance in the columns
#
# metric appropriateness: there is some imbalance in the data, perhaps accuracy is not appropriate. 
#                         Here it would depend a lot on the precise question we want to answer.
#                         objectives like "predict if the student got a passing grade" are common
#                         but in practice it is often too vague. to decide on a metric
#
## One Hot Encoding of all the categorical features
XOH = pd.get_dummies(X,drop_first=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XOH,y,stratify=y)

y_test.value_counts()

## solving the problem of leakage by comparing models on the test set
## with a simple cross-validation scheme:
from sklearn.model_selection import cross_val_score

CVscore_LR =  cross_val_score( LogisticRegression(max_iter=10**3) , 
                              X_train, y_train , 
                              cv=5 , scoring='accuracy' ).mean()
CVscore_KNN = cross_val_score( KNeighborsClassifier() , 
                              X_train, y_train , 
                              cv=5 , scoring='accuracy' ).mean()

print( "default Logistic regression cv-accuracy:" , CVscore_LR )
print( "default KNN cv-accuracy:" , CVscore_KNN )


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

pipeline_lr=Pipeline([('scalar',StandardScaler()), 
                      ('model',LogisticRegression(solver='liblinear',
                                                  n_jobs=1)  )])
grid_values = {'model__C': np.logspace(-2,2,100),
               'model__penalty': ['l1','l2'] 
              }
grid_LR = GridSearchCV( pipeline_lr,
                           param_grid = grid_values,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                         )
grid_LR.fit(X_train, y_train)

print(f'Grid best score ({grid_LR.scoring}): {grid_LR.best_score_:.3f}')
print('Grid best parameter :')

for k,v in grid_LR.best_params_.items():
    print('{:>25}\t{}'.format(k,v))

from sklearn.neighbors import KNeighborsClassifier
pipeline_knn=Pipeline([('scalar',StandardScaler()), 
                      ('model', KNeighborsClassifier()  )])
grid_values = {'model__n_neighbors': np.arange(1,50),
               'model__weights': ['uniform','distance']  
              }
grid_KNN = GridSearchCV( pipeline_knn,
                           param_grid = grid_values,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1
                         )
grid_KNN.fit(X_train, y_train)
print(f'Grid best score ({grid_KNN.scoring}): {grid_KNN.best_score_:.3f}')
print('Grid best parameter :')

for k,v in grid_KNN.best_params_.items():
    print('{:>25}\t{}'.format(k,v))

## Now, we select the LR model based on cross-validated accuracy
##      and see how it does on the test set
from sklearn.metrics import classification_report

y_pred = grid_LR.best_estimator_.predict(X_test)

sns.heatmap( pd.crosstab( y_test , y_pred ) , annot=True, fmt=".0f" )
plt.xlabel('predictions')
plt.ylabel('truth')


print( classification_report( y_test , y_pred ) )
