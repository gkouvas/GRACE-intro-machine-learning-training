from sklearn.tree import plot_tree
fig,ax = plt.subplots(figsize=(32,4))

_ = plot_tree( grid_tree.best_estimator_ , feature_names=X_train.columns , label='root',
               fontsize=8 , filled=True , impurity=False , precision=1, ax=ax)
