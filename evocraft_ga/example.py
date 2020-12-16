import grpc

import evocraft_ga.external.minecraft_pb2_grpc as minecraft_pb2_grpc
from evocraft_ga.external.minecraft_pb2 import *  # noqa

channel = grpc.insecure_channel("localhost:5001")
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)
x = 20
y = 10
z = 20
client.fillCube(
    FillCubeRequest(  # Clear a 20x10x20 working area
        cube=Cube(min=Point(x=x-10, y=y, z=z-10), max=Point(x=x+10, y=y+10, z=z+10)),
        type=AIR,
    )
)
client.spawnBlocks(
    Blocks(
        blocks=[  # Spawn a flying machine
            # Lower layer
            Block(
                position=Point(x=1 + x, y=5 + y, z=1 + z),
                type=PISTON,
                orientation=NORTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=0 + z), type=SLIME, orientation=NORTH
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=-1 + z),
                type=STICKY_PISTON,
                orientation=SOUTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=-2 + z),
                type=PISTON,
                orientation=NORTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=-4 + z),
                type=SLIME,
                orientation=NORTH,
            ),
            # Upper layer
            Block(
                position=Point(x=1 + x, y=6 + y, z=0 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            Block(
                position=Point(x=1 + x, y=6 + y, z=-4 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            # Activate
            Block(
                position=Point(x=1 + x, y=6 + y, z=-1 + z),
                type=QUARTZ_BLOCK,
                orientation=NORTH,
            ),
        ]
    )
)


blocks = client.readCube(Cube(min=Point(x=1, y=5, z=-4), max=Point(x=1, y=6, z=1)))

print(blocks)
