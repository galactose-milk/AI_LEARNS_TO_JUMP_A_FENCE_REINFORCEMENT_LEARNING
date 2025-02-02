import numpy as np
import torch
import torch.nn as nn
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

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
        self.camera.setPos(-5, -12, 3)
        self.camera.lookAt(5, 0, 0)
        
        # Physics parameters
        self.gravity = -9.81
        self.jump_force = 7.0
        self.forward_speed = 2.0
        self.dt = 0.05
        self.fence_position = 5.0
        self.fence_height = 1.5
        self.restitution = 0.5  # Bounce factor
        self.friction = 0.3  # Friction coefficient
        
        # Agent state
        self.position = np.array([-2.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Create the ground
        self.ground = self.loader.loadModel("models/environment")
        self.ground.setScale(20, 10, 1)
        self.ground.setPos(0, 0, -0.1)
        ground_material = Material()
        ground_material.setDiffuse((0.2, 0.6, 0.2, 1))
        self.ground.setMaterial(ground_material)
        self.ground.reparentTo(self.render)
        
        # Create fences with collision boxes
        self.fences = []
        fence_positions = [(1, -3.8), (3, -3.8)]
        
        for pos_x, pos_y in fence_positions:
            fence = self.loader.loadModel("models/box")
            fence.setScale(0.2, 2, 1)
            fence.setPos(pos_x, pos_y, self.fence_height / 2)
            
            fence_material = Material()
            fence_material.setDiffuse((0.8, 0.4, 0.2, 1))
            fence_material.setAmbient((0.2, 0.1, 0.05, 1))
            fence_material.setSpecular((1, 1, 1, 1))
            fence_material.setShininess(30)
            fence.setMaterial(fence_material)
            fence.reparentTo(self.render)
            
            # Create collision box
            fence_col = CollisionBox(Point3(-0.1, -1, -0.5),
                                   Point3(0.1, 1, 0.5))
            fence_col_node = CollisionNode(f'fence_{len(self.fences)}')
            fence_col_node.addSolid(fence_col)
            fence_col_np = fence.attachNewNode(fence_col_node)
            
            self.fences.append({
                'model': fence,
                'collision': fence_col_np,
                'position': np.array([pos_x, pos_y, self.fence_height / 2])
            })
        
        # Create the agent with collision sphere
        self.agent = self.loader.loadModel("smiley.egg.pz")
        self.agent.setScale(0.3, 0.3, 0.3)
        agent_material = Material()
        agent_material.setDiffuse((1, 0.2, 0.2, 1))
        agent_material.setEmission((0.5, 0.1, 0.1, 1))
        self.agent.setMaterial(agent_material)
        self.agent.setPos(*self.position)
        self.agent.reparentTo(self.render)
        
        # Create collision sphere for agent
        self.agent_radius = 0.3
        agent_col = CollisionSphere(0, 0, 0, self.agent_radius)
        agent_col_node = CollisionNode('agent')
        agent_col_node.addSolid(agent_col)
        self.agent_col_np = self.agent.attachNewNode(agent_col_node)
        
        # Set up collision traverser and handler
        self.cTrav = CollisionTraverser()
        self.collHandler = CollisionHandlerQueue()
        self.cTrav.addCollider(self.agent_col_np, self.collHandler)
        
        # Add path markers
        self.add_path_markers()
        self.setup_lighting()
        self.taskMgr.add(self.update, "update")
        
    def add_path_markers(self):
        for x in range(-2, 8, 1):
            marker = self.loader.loadModel("models/box")
            marker.setScale(0.05, 0.5, 0.02)
            marker.setPos(x, 0, 0)
            marker_material = Material()
            marker_material.setDiffuse((1, 1, 1, 1))
            marker.setMaterial(marker_material)
            marker.reparentTo(self.render)
        
    def setup_lighting(self):
        render.setShaderAuto()
        
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -60, 0)
        self.render.setLight(dlnp)
        
        self.plight = PointLight('plight')
        self.plight.setColor((0.6, 0.6, 0.6, 1))
        self.plight.setAttenuation((1, 0, 0.5))
        self.plnp = self.render.attachNewNode(self.plight)
        self.render.setLight(self.plnp)
        
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
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
        
    def handle_collisions(self):
        for entry in self.collHandler.getEntries():
            # Get collision normal
            normal = entry.getSurfaceNormal(render)
            normal_vec = np.array([normal.getX(), normal.getY(), normal.getZ()])
            
            # Get current velocity vector
            velocity_vec = np.array([self.velocity[0], 0, self.velocity[1]])
            
            # Reflect velocity about normal with restitution
            reflection = velocity_vec - 2 * np.dot(velocity_vec, normal_vec) * normal_vec
            reflection *= self.restitution
            
            # Apply friction to horizontal component
            reflection[0] *= (1 - self.friction)
            
            # Update velocity
            self.velocity[0] = reflection[0]
            self.velocity[1] = reflection[2]
            
            # Move agent slightly away from collision point to prevent sticking
            push_distance = 0.1
            self.position += normal_vec[:3] * push_distance
        
    def reset(self):
        self.position = np.array([-2.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.agent.setPos(*self.position)
        
    def update(self, task):
        # Update point light
        self.plnp.setPos(self.position[0], -2, 3)
        
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
        old_position = self.position.copy()
        self.position[0] += self.velocity[0] * self.dt
        self.position[1] += self.velocity[1] * self.dt
        
        # Ground collision
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = 0
            self.velocity[0] *= (1 - self.friction)  # Apply friction when on ground
        
        # Update agent position
        self.agent.setPos(*self.position)
        
        # Check collisions
        self.cTrav.traverse(render)
        self.handle_collisions()
        
        # Reset if too far
        if self.position[0] > self.fence_position + 2:
            self.reset()
        
        return task.cont

if __name__ == "__main__":
    visualizer = JumpVisualizer()
    visualizer.run()