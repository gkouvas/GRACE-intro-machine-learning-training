from sklearn.tree import plot_tree
fig,ax = plt.subplots(figsize=(25,10))
plot_tree( grid_tree_roc_auc.best_estimator_ , 
          feature_names=df_cancer.columns[:-1] , 
          ax=ax , fontsize=12 , filled=True , impurity=False , precision=3)
ax.set_title('best single decision tree')


## We can see that the first node splits on mean_concave_point,
## then on mean_perimeter and mean_texture
## we also al see that below the 2nd layer the nodes don't make the decision change for a threshold of 0.05
## this can still be useful, especially when we have metrics which accounts for multiple thresholds
## such as ROC AUC
##
## but sometimes it can be interesting to "prune" the useless nodes. See:
## https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
