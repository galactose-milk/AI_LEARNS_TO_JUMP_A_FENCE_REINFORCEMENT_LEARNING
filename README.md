# AI Fence Jumper

## Overview
This project utilizes **Reinforcement Learning (RL)** to train an AI agent to jump over fences in a simulated environment. The implementation is built using **Pygame** for rendering and **Panda3D** for 3D visualization. The AI learns through trial and error, optimizing its jumping strategy over time.

## Features
- **Reinforcement Learning:** Uses a deep Q-learning approach to train the AI.
- **Pygame Integration:** Handles basic physics, game loop, and environment interactions.
- **Panda3D Visualization:** Provides a 3D representation of the environment for better insights.
- **Customizable Training:** Adjustable parameters for learning rate, reward system, and jump mechanics.
- **Performance Logging:** Tracks AI performance and training progress over time.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8) and install the required dependencies.

```sh
pip install pygame panda3d numpy matplotlib torch
```

## Usage
1. **Run the training script:**
   ```sh
   python train.py
   ```
   This will initialize the environment and start training the AI.

2. **Test the trained model:**
   ```sh
   python test.py
   ```
   This allows you to see how well the AI performs after training.

3. **Visualize results:**
   ```sh
   python visualize.py
   ```
   This will open a Panda3D window showing the AI's movements in 3D.

## How It Works
1. The AI receives input from the environment (such as fence position, speed, and height).
2. It decides whether to jump or keep running based on a trained policy.
3. The agent receives rewards for successful jumps and penalties for failures.
4. Over multiple training iterations, the AI optimizes its actions to maximize successful jumps.

## Training Details
- **State Representation:** Position, velocity, fence distance, and jump height.
- **Action Space:** Jump or continue running.
- **Reward System:**
  - +10 for successfully jumping a fence.
  - -5 for hitting the fence.
  - +1 for moving forward.
- **Algorithm Used:** Deep Q-Network (DQN) with experience replay.

## Future Improvements
- Implement more complex obstacles.
- Fine-tune hyperparameters for better performance.
- Add different terrains and variable fence heights.

## Contributing
Feel free to fork the repository and submit pull requests for improvements!

## License
This project is licensed under the GNU Standard License. See `LICENSE` for more details.

## Author
Sanskar Kulkarni


