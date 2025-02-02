import numpy as np
import torch
import torch.nn as nn
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

# DQN Network (same as before)
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
        
        # Set up the camera
        self.disableMouse()
        self.camera.setPos(0, -15, 5)
        self.camera.lookAt(0, 0, 0)
        
        # Environment parameters
        self.gravity = -9.81
        self.jump_force = 7.0
        self.forward_speed = 2.0
        self.dt = 0.05
        self.fence_position = 5.0
        self.fence_height = 1.5
        
        # Agent state
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Create the ground
        self.ground = self.loader.loadModel("models/environment")
        self.ground.setScale(10, 10, 1)
        self.ground.setPos(0, 0, 0)
        self.ground.reparentTo(self.render)
        
        # Create the fence with material and color
        self.fence = self.loader.loadModel("models/box")
        self.fence.setScale(0.1, 2, self.fence_height)  # Made wider for better visibility
        self.fence.setPos(self.fence_position, 0, self.fence_height / 2)
        
        # Add material to fence
        fence_material = Material()
        fence_material.setDiffuse((0.8, 0.4, 0.2, 1))  # Brown color
        fence_material.setAmbient((0.2, 0.1, 0.05, 1))
        fence_material.setSpecular((1, 1, 1, 1))
        fence_material.setShininess(30)
        self.fence.setMaterial(fence_material)
        
        # Apply a texture to make the fence more visible
        cm = CardMaker("fence_card")
        fence_tex = self.loader.loadTexture("maps/noise.rgb")  # Use noise texture for wood-like appearance
        self.fence.setTexture(fence_tex)
        
        self.fence.reparentTo(self.render)
        
        # Create the agent (a sphere)
        self.agent = self.loader.loadModel("models/smiley")
        self.agent.setScale(0.3, 0.3, 0.3)
        self.agent.setPos(*self.position)
        self.agent.reparentTo(self.render)
        
        # Set up enhanced lighting
        self.setup_lighting()
        
        # Start the update task
        self.taskMgr.add(self.update, "update")
        
    def setup_lighting(self):
        # Enable per-pixel lighting
        render.setShaderAuto()
        
        # Main directional light (sun-like)
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -45, 0)  # Adjust light direction
        self.render.setLight(dlnp)
        
        # Point light for additional illumination
        plight = PointLight('plight')
        plight.setColor((0.4, 0.4, 0.4, 1))
        plight.setAttenuation((1, 0, 0.5))  # Adjust light falloff
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(0, -10, 10)
        self.render.setLight(plnp)
        
        # Ambient light to prevent completely dark shadows
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
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
        self.agent.setPos(*self.position)
        
    def update(self, task):
        # Get model prediction
        state = self._get_observation()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = q_values.max(1)[1].item()
        
        # Apply action
        if action == 1 and self.position[1] == 0:
            self.velocity[1] = self.jump_force
        
        # Update physics
        self.velocity[1] += self.gravity * self.dt
        self.position[0] += self.forward_speed * self.dt
        self.position[1] += self.velocity[1] * self.dt
        
        # Ground collision
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = 0
        
        # Update agent position
        self.agent.setPos(self.position[0], 0, self.position[1])
        
        # Check if episode should reset
        if self.position[0] > self.fence_position + 2:
            self.reset()
        
        return task.cont

if __name__ == "__main__":
    visualizer = JumpVisualizer()
    visualizer.run()