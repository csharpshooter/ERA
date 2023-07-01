## ERA Assignment 9
-------------------
## Name : Abhijit Mali
----------------------
## Notes
---------------------------------------------------------------------------------------------------------------------------
1) Used GPU
2) Achieved 83.010000 test accuracy. Model is overfitting.
3) Receptive Field = 91
4) Link to dephtwise seperable convolution class : https://github.com/csharpshooter/ERA/blob/main/A9/src/models/depthwise_seperable_conv2d.py
5) Used Depthwise Separable Convolution
6) Link to cnn model : (https://github.com/csharpshooter/ERA/blob/main/A9/src/models/cnn_model.py)
7) used Dialated convolution with stride of 2 instead of max pooling in last layer of block 
8) used FC after GAP
9) Trained for 200 epochs, highest accuracy = 77
10) Implemented modular code, model checkpoint to save best model and also to save model along with loss and accuracy data
11) Implemented Image Augmentation using albumentations (Horizontal Flip, Coarse Dropout, Shift Scale Rotate)
12) Used less than 136928 parameters (less than 200k) https://github.com/csharpshooter/ERA/blob/main/A9/images/modelsummary.png
13) Link to python notebook: https://github.com/csharpshooter/ERA/blob/main/A9/A9.ipynb
---------------------------------------------------------------------------------------------------------------------------

## Project Structure
--------------------

![Project Structure](https://github.com/csharpshooter/ERA/blob/main/A9/images/ProjectStructure.png)

---------------------------------------------------------------------------------------------------------------------------
## Test and Train, Loss and Accuracy Graphs

![Graphs](https://github.com/csharpshooter/ERA/blob/main/A9/images/traintestgraphs.png)

---------------------------------------------------------------------------------------------------------------------------
## Misclassified Images Graph

![Misclassified](https://github.com/csharpshooter/ERA/blob/main/A9/images/missclassifiedimages.png)

---------------------------------------------------------------------------------------------------------------------------
## Model Summary

![ModelSummary](https://github.com/csharpshooter/ERA/blob/main/A9/images/modelsummary.png)

---------------------------------------------------------------------------------------------------------------------------
## Logs of last few Epochs

Learning rate = 0.002392993292306176  for epoch:  195
EPOCH: 195
Loss=0.7626842856407166 Batch_id=390 Accuracy=75.88: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:28<00:00, 13.83it/s]
Test set: Average loss: 0.0041, Accuracy: 8272/10000 (82.72%)

Learning rate = 0.002392993292306176  for epoch:  196
EPOCH: 196
Loss=0.647508978843689 Batch_id=390 Accuracy=75.70: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:28<00:00, 13.80it/s]
Test set: Average loss: 0.0041, Accuracy: 8261/10000 (82.61%)

Learning rate = 0.002392993292306176  for epoch:  197
EPOCH: 197
Loss=0.6867589950561523 Batch_id=390 Accuracy=75.61: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:28<00:00, 13.75it/s]
Test set: Average loss: 0.0041, Accuracy: 8261/10000 (82.61%)

Learning rate = 0.002392993292306176  for epoch:  198
EPOCH: 198
Loss=0.8203479647636414 Batch_id=390 Accuracy=75.63: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:27<00:00, 14.21it/s]
Test set: Average loss: 0.0041, Accuracy: 8280/10000 (82.80%)

Epoch 00198: reducing learning rate of group 0 to 2.1537e-03.
Learning rate = 0.0021536939630755585  for epoch:  199
EPOCH: 199
Loss=0.7468287348747253 Batch_id=390 Accuracy=76.02: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:26<00:00, 15.02it/s]
Test set: Average loss: 0.0040, Accuracy: 8301/10000 (83.01%)

**Validation accuracy increased (82.990000 --> 83.010000).  Saving model ...**
Learning rate = 0.0021536939630755585  for epoch:  200
EPOCH: 200
Loss=0.8069970011711121 Batch_id=390 Accuracy=75.66: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:26<00:00, 14.96it/s]
Test set: Average loss: 0.0042, Accuracy: 8242/10000 (82.42%)

Learning rate = 0.0021536939630755585  for epoch:  201
