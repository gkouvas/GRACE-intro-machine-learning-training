fig,ax = plt.subplots( 1,2,figsize=(30,15))

sns.heatmap(X_ph_meanI.corr(),ax=ax[0], cmap='jet' , vmin = 0 , vmax = 1)
ax[0].set_title(f"mean imputation : score = {get_avg_replicate_corr(X_ph_meanI):.3f}")

sns.heatmap(X_ph_knnI.corr(), ax= ax[1], cmap='jet' , vmin = 0 , vmax = 1)
ax[1].set_title(f"KNN imputation : score = {get_avg_replicate_corr(X_ph_knnI):.3f}")

fig.suptitle('sample correlation')
plt.tight_layout()
