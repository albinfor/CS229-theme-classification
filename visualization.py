import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
data_arraycnn = np.load('confusion_matrix_cnn.npy')
print(data_arraycnn)
data_arraydnn = np.load('confusion_matrix_dense.npy')
print(data_arraydnn)
data_arrayNB = np.load('confusion_matrix_NB.npy')
print(data_arrayNB)

data_arraycnn = normalize(data_arraycnn, axis=1, norm='l1')
data_arraydnn = normalize(data_arraydnn, axis=1, norm='l1')
data_arrayNB = normalize(data_arrayNB, axis=1, norm='l1')

data_arraycnn = np.around(data_arraycnn, decimals=2)
data_arraydnn = np.around(data_arraydnn, decimals=2)
data_arrayNB = np.around(data_arrayNB, decimals=2)


df_cm = pd.DataFrame(data_arraycnn, index = [i for i in "ABCDEFGHIJKKMNOP"],
                  columns = [i for i in "ABCDEFGHIJKLMNOP"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix CNN')
plt.savefig('cfCNN.pdf')

df_cm = pd.DataFrame(data_arraydnn, index = [i for i in "ABCDEFGHIJKKMNOP"],
                  columns = [i for i in "ABCDEFGHIJKLMNOP"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix DNN')
plt.savefig('cfDNN.pdf')

df_cm = pd.DataFrame(data_arrayNB, index = [i for i in "ABCDEFGHIJKKMNOP"],
                  columns = [i for i in "ABCDEFGHIJKLMNOP"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix NB')
plt.savefig('cfNB.pdf')