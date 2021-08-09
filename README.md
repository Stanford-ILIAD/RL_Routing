This code simulates a traffic network with mixed autonomy; trains an RL policy for the routing control of autonomous cars; and tests RL, selfish, greedy, and MPC policies.

Companion code to TRC paper:  
Daniel A. Lazar, Erdem Bıyık, Dorsa Sadigh, Ramtin Pedarsani. **"Learning how to Dynamically Route Autonomous Vehicles on Shared Roads"**. *Transportation Research Part C: Emerging Technologies*, vol. 130, pp. 103258, 2021; doi: 10.1016/j.trc.2021.103258.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [NumPy](https://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
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

### Greedy Controller
The greedy controller (based on \[20\] in the paper) can be run with
```python
	python greedy_optimization.py
```

### MPC-based Controller
We have finally implemented an MPC-based controller. For better compatibility, this is the version with no CPU parallelization.
```python
	python mpc_policy.py
```
