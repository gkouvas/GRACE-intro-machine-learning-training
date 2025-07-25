## importing stuff
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split , GridSearchCV, StratifiedKFold 
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

## looking at the data a bit 
print("TenYearCHD value counts:")
print( df_heart.TenYearCHD.value_counts() )
print("we can see there is a lot of imbalance")
print("\n***\n")
print("Fraction of NAs:")
print( df_heart.isna().mean() )
print("some NA, but not a huge amount")
print("\n***\n")
## splitting train and test
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(X_heart,y_heart,
                                                                            stratify=y_heart)
#stratify is here to make sure that you split keeping the repartition of labels unaffected

print(f"full : {sum(y_heart)} CHD / {len(y_heart)} samples")
print(f"train: {sum(y_heart_train)} CHD / {len(y_heart_train)} samples")
print(f"test : {sum(y_heart_test)} CHD / {len(y_heart_test)} samples")
print("\n***\n")
%%time
RF=RandomForestClassifier(class_weight="balanced",n_estimators=1000)

grid_values = {'max_depth':np.arange( 2,20,4 ),
               'min_samples_split': np.arange( 25,200,25 )}

grid_heart = GridSearchCV(RF, 
                         param_grid = grid_values, 
                         scoring="roc_auc",
                         cv = StratifiedKFold(n_splits=5 , shuffle=True, random_state=1234),
                         n_jobs=-1)

grid_heart.fit(X_heart_train, y_heart_train) #train your pipeline

print(f'Grid best score ({grid_heart.scoring}): {grid_heart.best_score_:.4f}')
print(f'Grid best parameter (max.{grid_heart.scoring}): ')
for p,v in grid_heart.best_params_.items():
    print('\t',p,'->',v)
