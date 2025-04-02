from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca(self):
    #PCA 主成份分析
    pca = PCA(n_components=2, iterated_power=1)
    train_reduced = pca.fit_transform(self.train_X)

    print('PCA方差比: ',pca.explained_variance_ratio_)
    print('PCA方差值:',pca.explained_variance_)

def tsne(self):
    #t-SNE 非線性的 t-隨機鄰近嵌入法
    tsneModel = TSNE(n_components=2, random_state=42,n_iter=1000)
    train_reduced = tsneModel.fit_transform(self.train_X)