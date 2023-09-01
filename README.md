Because Mnist is too simple, we exchange training sets and test sets to make it more difficult. All experiments are performed 5 times and we report the average of the accuracy.



[official implementation](https://github.com/HobbitLong/SupContrast/blob/master/losses.py):

|     Loss     | Test Accuracy |
| :----------: | :-----------: |
| CrossEntropy |    0.9783     |
|    SimCLR    |    0.9460     |
|    SupCon    |    0.9772     |



our implementation:

|     Loss     | Test Accuracy |
| :----------: | :-----------: |
| CrossEntropy |    0.9783     |
|    SimCLR    |    0.9475     |
|    SupCon    |    0.9769     |

