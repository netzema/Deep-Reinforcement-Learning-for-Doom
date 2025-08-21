# Deep Reinforcement Learning for Doom

This project applies deep reinforcement learning to train an agent for the game Doom. The agent learns combat strategies against automated bots in the VizdoomMPEnv using a Dueling Deep Q-Network (Dueling DQN).

## Objective

* Train an agent to play Doom effectively on the "ROOM" map against four bots.
* Learn visual and spatial features from raw game frames.
* Design a reward function that encourages exploration, movement, and effective combat.
* Evaluate agent performance both locally and on the challenge server.

## Methodology

* **Architecture**: Dueling DQN

  * Convolutional layers process grayscale 128×128 pixel frames.
  * Two fully connected streams estimate:

    * State value function V(s)
    * Action advantages A(s,a)
  * Combined to produce Q-value estimates.
* **Input**: Grayscale frames + depth buffer for distance information (no frame stacking due to errors).
* **Reward function**:

  * +100 per frag
  * +5 per successful hit
  * -0.1 for damage taken
  * -0.005 penalty for survival without engagement
  * Movement bonus: doubled hit reward, +20 added to frag if preceded by movement
* **Training**:

  * Replay buffer (10,000 → 20,000 samples)
  * Adam optimizer, learning rate 1e-4
  * Discount factor γ = 0.999
  * Batch size 32
  * Target network updated via EMA
  * ε-greedy exploration: ε decayed from 1.0 → 0.1 at rate 0.99997

## Results

* Reward shaping with movement incentives led to faster and more effective learning compared to the baseline reward setup.
* Best local model (episode 480):

  * Shaped return: 153.9
  * Evaluation shaped return: 308.8
* Challenge server score: **75 points**
* Training metrics:

  * Frag and hit counts increased steadily over episodes.
  * Movement bonuses learned progressively.
  * Q-loss decreased after initial instability, with Q-value predictions rising consistently.
* Some late-phase divergence observed: Q-values continued rising while actual returns partially declined, suggesting possible overestimation or policy drift.

**Takeaway**
Reward shaping proved critical: introducing incentives for movement prevented the agent from exploiting static “stand-and-shoot” strategies and led to more effective combat policies. The Dueling DQN successfully captured state-action advantages and reached competitive performance on the Doom environment.
