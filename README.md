# Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access

## Dependencies


1. python [link](https://www.python.org)
2. matplotlib [link](https://matplotlib.org/)
3. tensorflow > 1.0 [link](https://www.tensorflow.org/)
4. numpy [link](https://www.numpy.org/)
5. jupyter [link](http://jupyter.org/)

We recommend to install with [Anaconda](https://anaconda.org/anaconda/python) 


### To train the DQN ,run on terminal
```bash
git clone https://github.com/shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access.git
cd Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access
python train.py --type DRQN
python train.py --type DQN --with_per
```

To understand the code , I have provided jupyter notebooks:
1. How to use environment.ipynb
2. How to generate states.ipynb
3. How_to_create_cluster.ipynb

To run notebook,run on terminal
```bash
jupyter notebook
```
Default browser will open ipynb files. Run each command one by one


### The Lab Modes ( ref. SLM Lab)
https://slm-lab.gitbook.io/slm-lab/

We adopt SLM Lab way for a deep dive into the spec file.
Next, the lab mode specifies one of the modes used to run the lab:

- dev: for development with verbose logging, environment rendering, and helpful checks like gradient updates. This is slower but useful for development.
- train: for training an agent to completion. This disables the development helper tools and thus runs the fastest.
- train@{predir}: for resuming training, 
  - train@latest will use the latest run for a spec
  - train@data/reinforce_cartpole_2020_04_13_232521 will use the specified run.
- enjoy@{session_spec_file}: for replaying a trained model from a trial-session; session_spec_file specifies the spec file from a session, e.g. enjoy@data/reinforce_cartpole_2020_04_13_232521/reinforce_cartpole_t0_s0_spec.json.
- search: for running an experiment / hyperparameter search.

In Quick Start, we used the lab command to read the demo spec file at slm_lab/spec/demo.json, use the dqn_cartpole spec in it, and run the spec in dev mode. To rerun the demo in train mode, we can simply change the lab mode to train to get the following:

```
python run_lab.py slm_lab/spec/demo.json dqn_cartpole train

python run_lab.py {spec file} {spec name} {lab mode}

python run_main.py demo.json ppo_mbr train

python run_main.py demo.json ppo_mbr train@latest
```



This work is an inspiration from the paper
```
O. Naparstek and K. Cohen, “Deep multi-user reinforcement learning for dynamic spectrum access in multichannel wireless
networks,” to appear in Proc. of the IEEE Global Communications Conference (GLOBECOM), Dec. 2017
```




