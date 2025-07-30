from sklearn.metrics import silhouette_score

silhouettes = []

Ks = range( 2,10 )
for i,K in enumerate( Ks ):
    km = KMeans(n_clusters = K)
    km.fit( dfpca_ie ) 
    
    ## computing silhouette
    silhouettes.append( silhouette_score( dfpca_ie , km.labels_ )    )
    
# fitting Kmean with the K giving the best silhouette score
bestK = Ks[ np.argmax(silhouettes) ]
km = KMeans(n_clusters = bestK)
km.fit( dfpca_ie ) 


fig,ax = plt.subplots( 1,2 , figsize = (15,6) )
ax[0].plot( Ks, silhouettes )
ax[0].set_ylabel('silhouette')
ax[0].set_xlabel('K')
ax[0].grid(axis = 'x')


sns.scatterplot( dfpca_ie ,x = "PC0" , y = "PC1", hue = km.labels_, ax = ax[1])
