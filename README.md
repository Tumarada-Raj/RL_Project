# Reinforcement Learning course project of RAJ MOHAN TUMARADA WiSe 2022/23.

This project is an implementation of an RL-based Agent designed to solve Karel's tasks implemented in PyTorch. 
It contains a policy gradient algorithm variant called Actor-Critic algorithm to train a neural policy for solving Karel tasks. The final model is trained for 3,00,000 episodes

The project borrows ideas from the following githubs:

https://mengxinji.github.io/Blog/2019-04-08/Actor-Critic/
https://github.com/ShangtongZhang/DeepRL/tree/master/deep_rl

https://medium.com/geekculture/actor-critic-implementing-actor-critic-methods-82efb998c273
https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py

To train the policy please follow the following steps:
--------------------------------------------------------
1. Place the unzipped datasets directory within the project.

2. Run the following command on the terminal ---> python train.py

```

The result of the best performing model on the validation dataset is  

Base A2C Performance:
Attempted 2400 tasks, correctly solved 2323 of them. Accuracy(solved)=96.79%



The ./pretrained folder contains the trained model with name actor_critic_final_test1.pth


Deploy Policy:
-------------
The latest version contains an additional main.py which can be executed from command line to use the trained neural policy on unseen tasks. 
Also it has some changes w.r.t to saving and loading a pretrained policy.


To Deploy the trained policy please follow the following steps:
----------------------------------------------------------------
1. Place the unzipped datasets directory within the project.
2. Make sure to set the path in output 
3. Run the following command on the terminal with appropriate task location    
    a. python main.py -t <Path to task>  (To test on a specific task and the generated sequence will be displayed on the console)
    b. python main.py -d <Path to Dataset>  (To test on multiple tasks within a dataset,  make sure the test dataset folder is named as test_without_seq) \
                      -o <Path to output dir> (To save the generated seqs of the dataset in json format) \
                      -l <level of directory (easy, medium, hard)>

The generated seq. for test dataset were submitted separately.
		

