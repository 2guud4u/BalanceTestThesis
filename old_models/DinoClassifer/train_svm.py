import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "../")))

from dataloader import  get_dino_dataset, create_val_train_splits, get_label_mapping
features,labels, fps_list=get_dino_dataset("/code/jjiang23/pathml/aim2_balance/processed_files/train_set.txt")
print(f"Features shape: {features[0].shape}, Labels shape: {labels[0].shape}")
train_loader, val_loader, train_dataset, val_dataset, class_weights = create_val_train_splits(features,labels)
from sklearn import svm
import torch
all_features = []
all_labels = []

for batch in train_loader:
    features, labels = batch
    all_features.append(features)
    all_labels.append(labels)
print(f"Total features shape: {torch.cat(all_features).shape}, Total labels shape: {torch.cat(all_labels).shape}")
X = torch.cat(all_features).numpy()
y = torch.cat(all_labels).numpy()

clf = svm.SVC(gamma='scale', verbose=True, class_weight='balanced')
clf.fit(X, y)


# validation
all_labels = []
all_predictions = []
for batch in val_loader:
    features, labels = batch
    print(f"Validation batch features shape: {features.shape}, Validation batch labels shape: {labels.shape}")
    predictions = clf.predict(features)
    # Here you can add code to evaluate the predictions against the labels
    all_labels.extend(labels.numpy())
    all_predictions.extend(predictions)

# confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)
# Confusion Matrix:
# [[  223   830   226     0  2038]
#  [  533   893   574   151  2394]
#  [  231   889   445    54  2252]
#  [    3    26   134   653  1348]
#  [ 1532  1670   801   457 21296]]