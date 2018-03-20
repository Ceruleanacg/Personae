# mayu - RL & SL Methods and Environment For Trading
New frameworks and envs with implementations of RL and SL methods for trading.   
This project is trying to implement Papers in Deep Reinforcement Learning and Supervised Learning and applying them in Financial Market.

# Content

+ [Deep Deterministic Policy Gradient (DDPG)](algorithm/RL/DDPG.py)   
Implement of DDPG with TensorFlow.

+ [DA-RNN (DualAttnRNN)](algorithm/SL/DualAttnRNN.py)   
Implement of A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction with TensorFlow.
    > arXiv:1704.02971: [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)

# How to Use
If you use Docker, you can run it by:
```
docker run -t -v local_project_dir:docker_project_dir --network=yuor_network ceruleanwang/haru agent/agent_name.py
```
Before you run agent in Docker, be sure you have mongodb container running.   
Or you can just run code below to pull & run a mongo container:
```
docker run -p 27017:27017 -v /data/db:/data/db -d --network=your_network mongo
``` 
Then you should make sure you have stocks data in mongodb.   
If you don't have, that does not matter, you can use spider in this project to crawl stocks data by:   
```
docker run -t -v local_project_dir:docker_project_dir --network=your_network ceruleanwang/haru spider/finance.py
```
This is also need mongodb running, please make sure mongodb is running first.

# TODO
- More Implementations of Papers.
- More High-Frequency Stocks Data.
