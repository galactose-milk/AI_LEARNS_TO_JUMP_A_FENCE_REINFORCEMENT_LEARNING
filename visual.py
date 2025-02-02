import numpy as np
import torch
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import torch.nn as nn

# DQN Network (same architecture as training for loading)
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

class JumpVisualizer:
    def __init__(self):
        # Initialize PyGame and OpenGL
        pygame.init()
        self.display = (1200, 800)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Jump Visualization")
    
        # Enable depth testing and lighting
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])
    
        # Set clear color (light blue sky)
        glClearColor(0.529, 0.808, 0.922, 1.0)
    
        # Set up the camera
        gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, -2.0, -15.0)
        glRotatef(20, 1, 0, 0)  # Tilt view down slightly
        
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
        
        # Colors
        self.ground_color = (0.8, 0.8, 0.8, 1.0)
        self.agent_color = (1.0, 0.0, 0.0, 1.0)
        self.fence_color = (0.6, 0.3, 0.1, 1.0)
        self.grid_color = (0.7, 0.7, 0.7, 1.0)
        
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
    def _load_model(self):
        # Create model instance
        model = DQN(4, 2).to(self.device)
        # Load the trained weights
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
        
    def _draw_ground_grid(self):
        glBegin(GL_LINES)
        glColor4f(*self.grid_color)
        
        # Draw grid lines
        for i in range(-10, 11):
            # Lines along X axis
            glVertex3f(i, 0, -10)
            glVertex3f(i, 0, 10)
            # Lines along Z axis
            glVertex3f(-10, 0, i)
            glVertex3f(10, 0, i)
            
        glEnd()
        
    def _draw_fence(self):
        glColor4f(*self.fence_color)
        glPushMatrix()
        glTranslatef(self.fence_position, self.fence_height/2, 0)
        glScalef(0.1, self.fence_height, 2.0)
        self._draw_cube()
        glPopMatrix()
        
    def _draw_agent(self):
        glColor4f(*self.agent_color)
        glPushMatrix()
        glTranslatef(*self.position)
        
        # Draw sphere for agent
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.3, 32, 32)
        
        # Draw shadow
        glColor4f(0.2, 0.2, 0.2, 0.5)
        glPushMatrix()
        glTranslatef(0, -self.position[1], 0)
        glScalef(1, 0.1, 1)
        gluSphere(quadric, 0.3, 32, 32)
        glPopMatrix()
        
        glPopMatrix()
        
    def _draw_cube(self):
        vertices = [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1]
        ]
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,7], [7,6], [6,4],
            [0,4], [1,5], [2,7], [3,6]
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3f(*vertices[vertex])
        glEnd()
        
    def reset(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
    def update(self):
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
        
        # Check if episode should reset
        if self.position[0] > self.fence_position + 2:
            self.reset()
            
    def render(self):
        # Clear buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Reset camera position
        glTranslatef(0.0, -2.0, -15.0)
        glRotatef(20, 1, 0, 0)
        
        # Follow the agent with camera
        glTranslatef(-self.position[0], 0, 0)
        
        # Draw scene elements
        self._draw_ground_grid()
        self._draw_fence()
        self._draw_agent()
        
        # Update display
        pygame.display.flip()
        
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
            
            self.update()
            self.render()
            clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    visualizer = JumpVisualizer()
    visualizer.run()