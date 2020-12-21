import grpc
import numpy as np
import evocraft_ga.external.minecraft_pb2_grpc as minecraft_pb2_grpc
from evocraft_ga.external.minecraft_pb2 import *  # noqa

class Spawner:
    def __init__(self, start_x=20, start_y=10, start_z=20, cube_len=10, class_dict={0:AIR, 1:REDSTONE_BLOCK}, orientation=SOUTH):
        self.start_x = start_x
        self.start_y = start_y
        self.start_z = start_z
        self.cube_len = cube_len
        self.class_dict = class_dict
        self.class_dict[-1.0] = AIR
        self.orientation = orientation
        self.channel = grpc.insecure_channel("localhost:5001")
        self.client = minecraft_pb2_grpc.MinecraftServiceStub(self.channel)

    def create_block(self, x, y, z, block_type):
        return Block(
                position=Point(x=x, y=y, z=z),
                type=block_type,
                orientation=self.orientation,
            )  
    
    def clear_blocks(self, x_min, y_min, z_min):
        
        self.client.fillCube(
            FillCubeRequest(  # Clear a 20x10x20 working area
                cube=Cube(
                    min=Point(x=x_min, y=y_min, z=z_min), max=Point(x=x_min + self.cube_len, y=y_min + self.cube_len, z=z_min + self.cube_len)
                ),
                type=AIR,
            )
        )

    def _populate_arr(self, arr, x_min, y_min, z_min):
        blocks = []
        self.clear_blocks(x_min,y_min,z_min)
        for coord in np.ndindex(arr.shape):
            block = self.create_block(x=x_min+coord[0], y=y_min+coord[1], z=z_min+coord[2], block_type=self.class_dict[arr[coord]])
            blocks.append(block)
        blocks = Blocks(blocks=blocks)
        self.client.spawnBlocks(blocks)
    
    def populate(self, cube_probs):
        for i in range(len(cube_probs)):
            dist = i*5 + i*self.cube_len
            self._populate_arr(cube_probs[i], self.start_x+dist,self.start_y, self.start_z)

    def clear_population(self, population_size):
        for i in range(population_size):
            dist = i*5 + i*self.cube_len
            self.clear_blocks(self.start_x+dist,self.start_y, self.start_z)
    