from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

sns.histplot( x = y_train , binwidth=100 , alpha = 0.5 )
sns.histplot( x = y_test , binwidth=100 , alpha = 0.5 )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

pipeline_lr=Pipeline([('scalar',StandardScaler()), 
                      ('poly',PolynomialFeatures()),
                      ('model',SGDRegressor()  )])

grid_values = {'model__alpha': np.logspace(-2,2,100),
               'model__l1_ratio': np.linspace(0.0,1.0,11),
               'poly__degree': [1,2],
               'poly__interaction_only': [True,False]
              }
grid_LR = GridSearchCV( pipeline_lr,
                           param_grid = grid_values,
                           scoring='neg_root_mean_squared_error',
                           cv=10,
                           n_jobs=-1
                         )
grid_LR.fit(X_train, y_train)
print( grid_LR.best_score_ )
print( grid_LR.best_params_ )
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

grid_values = {'criterion': ['squared_error'],
               'n_estimators':[100], 
               'max_depth':[5,7,9,11,13],
               'min_samples_split':[2,4,8,12]}

grid_RF = GridSearchCV( RandomForestRegressor(),
                           param_grid = grid_values,
                           scoring='neg_root_mean_squared_error',
                           cv=10,
                           n_jobs=-1
                         )

grid_RF.fit(X_train, y_train)
print( grid_RF.best_score_ )
print( grid_RF.best_params_ )
from sklearn.metrics import r2_score,root_mean_squared_error
train_pred = grid_RF.predict(X_train)
test_pred  = grid_RF.predict(X_test)


print("Train:")
print(f"\tr2  : {r2_score(y_train,train_pred):.2f}")
print(f"\tRMSE: {root_mean_squared_error(y_train,train_pred):.2f}")
print("Test:")
print(f"\tr2  : {r2_score(y_test,test_pred):.2f}")
print(f"\tRMSE: {root_mean_squared_error(y_test,test_pred):.2f}")


sns.scatterplot( x=y_train , y=train_pred , label='train')
sns.scatterplot( x=y_test , y=test_pred , label='test')
plt.ylabel("prediction")
plt.legend()
FI = pd.DataFrame({'feature' : X.columns,
                   "weight" : grid_RF.best_estimator_.feature_importances_
                  }
                 )
print('Features sorted per importance in discriminative process')

FI.sort_values(by='weight' , ascending=False )
