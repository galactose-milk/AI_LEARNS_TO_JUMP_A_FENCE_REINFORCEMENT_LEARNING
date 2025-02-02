import numpy as np
import torch
import torch.nn as nn
from panda3d.core import PointLight
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class JumpVisualizer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        
        # Environment settings
        self.gravity = -9.81
        self.jump_force = 7.0
        self.forward_speed = 2.0
        self.dt = 0.05
        self.fence_position = 5.0
        self.fence_height = 1.5
        
        # Agent state
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Load models
        self.ground = self.loader.loadModel("models/plane")
        self.ground.setScale(20, 20, 1)
        self.ground.setPos(0, 0, 0)
        self.ground.reparentTo(self.render)
        
        self.fence = self.loader.loadModel("models/box")
        self.fence.setScale(0.1, 0.1, self.fence_height)
        self.fence.setPos(self.fence_position, 0, self.fence_height / 2)
        self.fence.reparentTo(self.render)
        
        self.agent = self.loader.loadModel("models/smiley")
        self.agent.setScale(0.3, 0.3, 0.3)
        self.agent.setPos(*self.position)
        self.agent.reparentTo(self.render)
        
        # Lighting
        self.setup_lighting()
        
        # Score system
        self.score = 0
        self.score_text = OnscreenText(text=f"Score: {self.score}", pos=(-0.9, 0.9), scale=0.07, fg=(1, 1, 1, 1))
        
        # Start tasks
        self.taskMgr.add(self.update, "update")
        self.taskMgr.add(self.update_camera, "update_camera")
        
        # Restart input
        self.accept("r", self.reset)
        
    def setup_lighting(self):
        plight = PointLight("plight")
        plight.setColor((1, 1, 1, 1))
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(0, -10, 10)
        self.render.setLight(plnp)
        
    def _load_model(self):
        model = DQN(4, 2).to(self.device)
        model.load_state_dict(torch.load('final_jump_model.pth', map_location=self.device))
        model.eval()
        return model
        
    def _get_observation(self):
        return np.array([
            self.position[0],
            self.position[1],
            self.fence_position - self.position[0],
            self.velocity[1]
        ])
        
    def reset(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.score = 0
        self.agent.setPos(*self.position)
        self.score_text.setText(f"Score: {self.score}")
        if hasattr(self, 'game_over_text'):
            self.game_over_text.destroy()
        
    def update(self, task):
        state = self._get_observation()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = q_values.max(1)[1].item()
        
        if action == 1 and self.position[1] == 0:
            self.velocity[1] = self.jump_force
        
        self.velocity[1] += self.gravity * self.dt
        self.position[0] += self.forward_speed * self.dt
        self.position[1] += self.velocity[1] * self.dt
        
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = 0
        
        self.agent.setPos(self.position[0], 0, self.position[1])
        self.score += 1
        self.score_text.setText(f"Score: {self.score}")
        
        if self.position[0] >= self.fence_position and self.position[1] < self.fence_height:
            self.show_game_over()
        
        return task.cont
    
    def show_game_over(self):
        self.game_over_text = OnscreenText(text="Game Over! Press R to Restart", pos=(0, 0), scale=0.1, fg=(1, 0, 0, 1))
        self.taskMgr.remove("update")
    
    def update_camera(self, task):
        self.camera.setPos(self.position[0] - 5, -10, 3)
        self.camera.lookAt(self.position[0], 0, 1)
        return task.cont

if __name__ == "__main__":
    visualizer = JumpVisualizer()
    visualizer.run()
