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
        self.restitution = 0.5
        self.friction = 0.3
        
        # Agent state
        self.position = np.array([-2.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # Create ground plane using CardMaker
        cm = CardMaker('ground')
        cm.setFrame(-10, 10, -5, 5)  # Create a 20x10 ground plane
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setP(-90)  # Rotate to lay flat
        self.ground.setZ(-0.1)  # Slightly below zero to prevent z-fighting
        
        # Ground material
        ground_material = Material()
        ground_material.setDiffuse((0.2, 0.6, 0.2, 1))  # Green color
        self.ground.setMaterial(ground_material)
        
        # Create fences with GeomNodes
        self.fences = []
        fence_positions = [(1, -3.8), (3, -3.8)]
        
        format = GeomVertexFormat.getV3n3c4()
        for pos_x, pos_y in fence_positions:
            # Create fence geometry
            vdata = GeomVertexData('fence', format, Geom.UHStatic)
            vertex = GeomVertexWriter(vdata, 'vertex')
            normal = GeomVertexWriter(vdata, 'normal')
            color = GeomVertexWriter(vdata, 'color')
            
            # Add vertices for a simple box
            def add_vertex(x, y, z, nx, ny, nz):
                vertex.addData3(x, y, z)
                normal.addData3(nx, ny, nz)
                color.addData4(0.8, 0.4, 0.2, 1)  # Brown color
            
            # Create box vertices
            s = 0.2  # scale
            h = 1.0  # height
            # Bottom face
            add_vertex(-s, -s, 0, 0, 0, -1)
            add_vertex(s, -s, 0, 0, 0, -1)
            add_vertex(s, s, 0, 0, 0, -1)
            add_vertex(-s, s, 0, 0, 0, -1)

# Top face
            add_vertex(-s, -s, h, 0, 0, 1)
            add_vertex(s, -s, h, 0, 0, 1)
            add_vertex(s, s, h, 0, 0, 1)
            add_vertex(-s, s, h, 0, 0, 1)

            
            # Create GeomTriangles
            prim = GeomTriangles(Geom.UHStatic)
            # Add indices for the box faces
            # Front face
            prim.addVertices(0, 1, 4)
            prim.addVertices(1, 5, 4)
            # Back face
            prim.addVertices(2, 3, 6)
            prim.addVertices(3, 7, 6)
            # Left face
            prim.addVertices(3, 0, 7)
            prim.addVertices(0, 4, 7)
            # Right face
            prim.addVertices(1, 2, 5)
            prim.addVertices(2, 6, 5)
            # Top face
            prim.addVertices(4, 5, 7)
            prim.addVertices(5, 6, 7)
            # Bottom face
            prim.addVertices(0, 1, 3)
            prim.addVertices(1, 2, 3)
            
            geom = Geom(vdata)
            geom.addPrimitive(prim)
            
            node = GeomNode('fence')
            node.addGeom(geom)
            
            fence = self.render.attachNewNode(node)
            fence.setPos(pos_x, pos_y, self.fence_height / 2)
            
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
        
        # Create the agent (ball) using a sphere geometry
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('ball', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        # Create sphere vertices
        radius = 1
        segments = 16
        rings = 16
        
        for i in range(rings + 1):
            v = i / rings
            phi = v * np.pi
            for j in range(segments):
                u = j / segments
                theta = u * 2 * np.pi
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                vertex.addData3(x, y, z)
                normal.addData3(x/radius, y/radius, z/radius)
                color.addData4(1, 0.2, 0.2, 1)  # Red color
        
        # Create triangles
        prim = GeomTriangles(Geom.UHStatic)
        for i in range(rings):
            for j in range(segments):
                next_j = (j + 1) % segments
                
                prim.addVertices(
                    i * segments + j,
                    i * segments + next_j,
                    (i + 1) * segments + j
                )
                prim.addVertices(
                    i * segments + next_j,
                    (i + 1) * segments + next_j,
                    (i + 1) * segments + j
                )
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode('ball')
        node.addGeom(geom)
        
        self.agent = self.render.attachNewNode(node)
        self.agent.setPos(*self.position)
        
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
        
        # Add path markers using CardMaker
        self.add_path_markers()
        self.setup_lighting()
        self.taskMgr.add(self.update, "update")
        
    def add_path_markers(self):
        for x in range(-2, 8, 1):
            cm = CardMaker('marker')
            cm.setFrame(-0.025, 0.025, 0, 0.5)
            marker = self.render.attachNewNode(cm.generate())
            marker.setP(-90)
            marker.setPos(x, 0, 0.01)  # Slightly above ground
            marker_material = Material()
            marker_material.setDiffuse((1, 1, 1, 1))
            marker.setMaterial(marker_material)
        
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

    # Rest of the methods remain the same as in your code
    def _load_model(self): ...  # Same as before
    def _get_observation(self): ...  # Same as before
    def handle_collisions(self): ...  # Same as before
    def reset(self): ...  # Same as before
    def update(self, task): ...  # Same as before

if __name__ == "__main__":
    visualizer = JumpVisualizer()
    visualizer.run()