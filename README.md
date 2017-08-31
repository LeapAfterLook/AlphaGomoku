AlphaGomoku

# Objective
- Adapt AlphaGo (Mastering the game of Go with deep neural networks and tree search) algorithm to Gomoku

# Reference
- AlphaGo: Mastering the game of Go with deep neural networks and tree search
- Convolutional and Recurrent Neural Network for Gomoku

# Function of Pakages
## util
- DatasetManager: parse gomoku records from dataset of renju.net
- GameManager: manage gomoku game (rule, win check)
- UserInterface: easily deal with board and action
## SLPN_CNN
the policy network (CNN model) trained by supervised learning
## RLPN_CNN
the policy network (CNN model) trained by reinforcement learning (REINFORCEMENT algorithm; a.k.a Monte-Carlo Policy Gradient)
