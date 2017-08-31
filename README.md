AlphaGomoku

# Objective
- Adapt AlphaGo (Mastering the game of Go with deep neural networks and tree search) algorithm to Gomoku

# Reference
- AlphaGo: Mastering the game of Go with deep neural networks and tree search
- Convolutional and Recurrent Neural Network for Gomoku

# Function of Pakages
<p2>- util </p2>
<tab>- DatasetManager: parse gomoku records from dataset of renju.net</tab>
<tab>- GameManager: manage gomoku game (rule, win check)</tab>
<tab>- UserInterface: easily deal with board and action</tab>
<p2>- SLPN_CNN: the policy network (CNN model) trained by supervised learning</p2>
<p2>- RLPN_CNN: the policy network (CNN model) trained by reinforcement learning (REINFORCEMENT algorithm; a.k.a Monte-Carlo Policy Gradient)</p2>
