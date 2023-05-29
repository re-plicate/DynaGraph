# DynaGraph
## model run
user could run main.py to easily start the training.
test.py is for evaluating the DynaGraph.
## 1.dataset
This project includes 2 widely used datasets PEMS04 and PEMS08 for reproducing the results.
## 2.Comparison with the newly referred algorithm
Due to some methods have no open-source code, it is hard to republicat and are marked with the marker *.

*represents the result is not fiiled by now but will be update continuously updated
![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Tab1.png)
![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Tab2.png)
## 3.Time cost on TAXIBJ dataset
![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Fig1.png)

Figure 1: One epoch training time results

![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Fig2.png)

Figure 2: Total training time results. We set the bacthsize of each model as 8 and record the time cost for final convergence. We define the final convergence of each model as that the accuracy is no longer improved in the next 10 epochs of learning, then we record the time cost between the training start and the best accuracy emerges as the total training time.
## 4.New added large-size dataset
