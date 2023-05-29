# DynaGraph
## model run
user could run main.py to easily start the training.
test.py is for evaluating the DynaGraph.
## 1.dataset
This project includes 3 widely used datasets PeMS04, PeMS08 and Q-Traffic for reproducing the results. Regarding the Q-Traffic dataset, it consists of three sub-datasets: query sub-dataset, traffic speed sub-dataset and road network sub-dataset. We conducted experiments on the traffic speed sub-dataset.
### Traffic Speed Sub-dataset
This sub-dataset was collected in Beijing, China between April 1, 2017 and May 31, 2017, from the [Baidu Map](https://map.baidu.com). This sub-dataset contains 15,073 road segments covering approximately 738.91 km. Figure 1 shows the spatial distribution of these road segments, respectively.
<div align=center>
<img src="https://github.com/JingqingZ/BaiduTraffic/blob/master/fig/beijing_road_seg_compressed.png"/>
</div>
<p align="center">Figure 1. Spatial distribution of the road segments in Beijing</p>
They are all in the 6th ring road (bounded by the lon/lat box of <116.10, 39.69, 116.71, 40.18>), which is the most crowded area of Beijing. The traffic speed of each road segment is recorded per minute. To make the traffic speed predictable, for each road segment, original authors use simple [moving average](https://en.wikipedia.org/wiki/Moving_average) with a 15-minute time window to smooth the traffic speed sub-dataset and sample the traffic speed per 15 minutes. Thus, there are totally 5856 (61×24×4) time steps, and each record is represented as road_segment_id, time_stamp ([0, 5856)) and traffic_speed (km/h).
There are some traffic speed samples as follows:
···
15257588940, 0, 42.1175  

..., ..., ...  
  
15257588940, 5855, 33.6599  

1525758913, 0, 41.2719  

..., ..., ...  

## 2.Comparison with the newly referred algorithm
Due to some methods have no open-source code, it is hard to republicat and are marked with the marker *.

*represents the result is not fiiled by now but will be continuously updated

![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Tab1.png)

The results show in the Table 1 demonstrate the comparable performance of DynaGraph. DynaGraph outperforms the baselines except for the Detectornet. However, this baseline has no open-source code and we refer the results from the initial paper.

![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Tab2.png)

The results in Table 2 compare DynaGraph with all the open-source baselines. It is surprising that DynaGraph achieve SOTA in both two datasets. STAWnet shows the closest performance of DynaGraph, which has some similar mechanism with DynaGraph.
## 3.Time cost on TAXIBJ dataset
![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Fig1.png)

Figure 1: One epoch training time results

![image](https://github.com/re-plicate/DynaGraph/blob/main/Fig/Fig2.png)

Figure 2: Total training time results. We set the bacthsize of each model as 8 and record the time cost for final convergence. We define the final convergence of each model as that the accuracy is no longer improved in the next 10 epochs of learning, then we record the time cost between the training start and the best accuracy emerges as the total training time.
## 4.New added large-size dataset
