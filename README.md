This code simulates a traffic network with P parallel roads, trains an RL policy for the routing control of autonomous cars, and tests RL and selfish policies.

Companion code to NeurIPS 2019 submission:  
Anonymous Authors. **"Learning how to Dynamically Route Autonomous Vehicles on Shared Roads"**. _Submitted to_ *2019 Conference on Neural Information Processing Systems (NeurIPS)*, Vancouver, Canada, Dec. 2019.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [NumPy](https://www.numpy.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [OpenAI Gym](https://gym.openai.com)
- [OpenAI Baselines](https://github.com/openai/baselines)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)


## Running
We have already included a few pretrained models. Therefore, you can directly test them without any training.
Refer to the publication for the parameters and the values we used.
Parameters can be set inside the runner python files.

### Training
You simply run
```python
	python train.py
```
This can take a lot of time as it will simulate 40 million time steps. This can be changed in train.py

### RL Policy
You can try (pre)trained policies by simply executing
```python
	python rl_policy.py
```

### Selfish Policy
Similarly, you can see what happens when all vehicles are selfish by running
```python
	python allselfish_policy.py
```
