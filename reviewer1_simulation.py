from sklearn.metrics import roc_auc_score
import pandas as pd


###Translation of matlab code from reviewer 1.
#### matlab :(
# data(1,:) = [0 0.2 0.5 0.8 1];
# data(2,:) = [0 4 10 16 20];
# data(3,:) = [20 16 10 4 0];
# ref = [];
# pre = [];
# for i = 1:5
#   pre = [pre; zeros(data(2,i) + data(3,i),1) + data(1,i)];
#   if data(2, i) > 0
#     ref = [ref; ones(data(2,i),1)];
#   end
#   if data(3, i) > 0
#     ref = [ref; zeros(data(3,i),1)];
#   end
# end
#######
####direct python translation from chatgpt (amended to be more readable).
scores = np.array([0, 0.2, 0.5, 0.8, 1])
values_balanced = np.array([[0, 4, 10, 16, 20],
                             [20, 16, 10, 4, 0]], dtype = int)
values_imbalanced = np.array([[0, 20, 10, 16, 20],
                              [500, 80, 10, 4, 0]], dtype = int)
scores_balanced = np.array([])
labels_balanced = np.array([])
scores_imbalanced = np.array([])
labels_imbalanced = np.array([])

for i in range(5): 
    ## this iterates thru each index in the array 
    ##+ appends array of size n_pos+n_neg with each
    #value equal to the score for that class.
    scores_balanced = np.append(scores_balanced, np.zeros(values_balanced[0, i] + values_balanced[1, i]) + scores[i])
    scores_imbalanced = np.append(scores_imbalanced, np.zeros(values_imbalanced[0, i] + values_imbalanced[1, i]) + scores[i])
    if values_balanced[0, i] > 0:
        labels_balanced = np.append(labels_balanced , np.ones(values_balanced[0, i]))
    if values_imbalanced[0, i] > 0:
        labels_imbalanced = np.append(labels_imbalanced, np.ones(values_imbalanced[0,i]))
    if values_balanced[1, i] > 0:
        labels_balanced = np.append(labels_balanced, np.zeros(values_balanced[1, i]))
    if values_imbalanced[1, i] > 0:
        labels_imbalanced = np.append(labels_imbalanced, np.zeros(values_imbalanced[1,i]))
df = pd.concat([pd.DataFrame({'Score':scores_balanced,'Label':labels_balanced}).assign(Type="Balanced"), pd.DataFrame({'Score':scores_imbalanced, 'Label':labels_imbalanced}).assign(Type="Imbalanced")])
roc_aucs = df.groupby('Type').apply(lambda x:roc_auc_score(x.Label, x.Score))
av_scores = df.groupby('Type').apply(lambda x:x.groupby('Label')['Score'].mean())


### Better code python supremacy 
scores_balanced = np.hstack([scores[i]+np.zeros(values_balanced[0,i]+values_balanced[1,i]) for i in range(5)])
scores_imbalanced = np.hstack([scores[i]+np.zeros(values_imbalanced[0,i]+values_imbalanced[1,i]) for i in range(5)])
labels_balanced = np.hstack([np.hstack([np.ones(values_balanced[0,i]), np.zeros(values_balanced[1,i])]) for i in range(5)])
labels_imbalanced = np.hstack([np.hstack([np.ones(values_imbalanced[0,i]), np.zeros(values_imbalanced[1,i])]) for i in range(5)])
df = pd.concat([pd.DataFrame({'Score':scores_balanced,'Label':labels_balanced}).assign(Type="Balanced"), pd.DataFrame({'Score':scores_imbalanced, 'Label':labels_imbalanced}).assign(Type="Imbalanced")])
roc_aucs = df.groupby('Type').apply(lambda x:roc_auc_score(x.Label, x.Score))
av_scores = df.groupby('Type').apply(lambda x:x.groupby('Label')['Score'].mean())



