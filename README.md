# Description
### Neural Network that learns to play CartPole-v1 from OpenAI gym

CartPole-v1 is a game where there is a pole attached by an un-actuated joint to a cart,
which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over,
moving it only left and right.
https://gym.openai.com/envs/CartPole-v1/

# Installation
## Dependencies
The program depends on the following libraries:

- OpenAI Gym

- Tensorflow (>=1.9.0)

- TfLearn

For detailed steps to install Tensorflow, follow the Tensorflow installation instructions. A typical user can install Tensorflow using one of the following commands:

```
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user gym
pip install --user tflearn
```

## Testing the installation and running the program

You can test that you have correctly installed the dependencies by running the following command:

```
python game_AI.py
```

If everything is correctly configured, the program shuld start and you should see how it's strategy it imporves in time, form epoch to epcoh, by checking the loss and the average score.

# Playing Improvement


Initalized the Neural Network using a set of completely random moves, which acted really poor, the average score on 4000 games being only 60.

During 3 epochs of training and 415 training steps, it's accuracy slowly increased, reaching a high of 61% and an average score of 212.78:


![](https://github.com/chiriacandrei25/Game-playing-AI/blob/master/Capture2.PNG)

