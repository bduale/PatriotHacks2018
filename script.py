# python 3.6.4

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# prepare data
df = pd.read_csv('inputs_final.csv')
df = df + 1
df[df < 0] = 0
x = np.array([i[0:781] for i in df.values])
y = np.array([i[781] for i in pd.read_csv('inputs_final.csv').values])

var_names = pd.read_csv('hackathon_els_variables.csv')
var_names = dict(zip(
	[i[0] for i in var_names.values],
	[i[1] for i in var_names.values]))

# feature selection - determine best features
x_orig = x
y_orig = y
anova = SelectKBest(f_classif, k=15)
anova.fit(x_orig, y_orig)
print("Best variables (no order):", 
	*[var_names[df.columns.get_values()[i]] for i in anova.get_support(True)], sep="\n - ")

# model selection

# LDA
print('-------------------------------------------------')
print("Training LDA...")
x_orig = x
y_orig = y
lda = LinearDiscriminantAnalysis()
lda.fit(x_orig, y_orig)
cs = cross_val_score(lda, x_orig, y_orig, cv=5)
print("Average LDA Accuracy:", '{:.2%}'.format(cs.mean()))

# MLP
print('-------------------------------------------------')
print("Training MLP...")
x_orig = x
y_orig = y
mlp = MLPClassifier(hidden_layer_sizes=(300,150))
mlp.fit(x_orig, y_orig)
cs = cross_val_score(mlp, x_orig, y_orig, cv=5)
print("Average MLP Accuracy:", '{:.2%}'.format(cs.mean()))

# LDA with ANOVA
print('-------------------------------------------------')
print("Training LDA with ANOVA...")
x_orig = x
y_orig = y
pipe = Pipeline([('anova', SelectKBest(f_classif, k=50)),
                ('lda', LinearDiscriminantAnalysis())])
pipe.fit(x_orig, y_orig)
pipe.score(x_orig, y_orig)
cs = cross_val_score(pipe, x_orig, y_orig, cv=5)
print("Average ANOVA LDA Accuracy:", '{:.2%}'.format(cs.mean()))

# MLP with ANOVA
print('-------------------------------------------------')
print("Training MLP with ANOVA...")
x_orig = x
y_orig = y
pipe = Pipeline([('anova', SelectKBest(f_classif, k=50)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(10,5)))])
pipe.fit(x_orig, y_orig)
pipe.score(x_orig, y_orig)
cs = cross_val_score(pipe, x_orig, y_orig, cv=5)
print("Average ANOVA MLP Accuracy:", '{:.2%}'.format(cs.mean()))

# K-Nearest Neighbors with ANOVA
print('-------------------------------------------------')
print("Training K-Nearest Neighbors with ANOVA...")
x_orig = x
y_orig = y
pipe = Pipeline([('anova', SelectKBest(f_classif, k=15)),
                ('knn', KNeighborsClassifier(n_neighbors = 300, 
                                             weights='distance'))])
pipe.fit(x_orig, y_orig)
cs = cross_val_score(pipe, x_orig, y_orig, cv=5)
print("Average ANOVA KNN Accuracy:", '{:.2%}'.format(cs.mean()))


