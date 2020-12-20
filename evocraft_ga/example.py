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
        cube=Cube(
            min=Point(x=x - 10, y=y, z=z - 10), max=Point(x=x + 10, y=y + 10, z=z + 10)
        ),
        type=AIR,
    )
)
client.spawnBlocks(
    Blocks(
        blocks=[  # Spawn a flying machine
            # Lower layer
            Block(
                position=Point(x=1 + x, y=5 + y, z=1 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=2 + z), type=REDSTONE_BLOCK, orientation=NORTH
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=3 + z),
                type=REDSTONE_BLOCK,
                orientation=SOUTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=4 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=5 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            # Upper layer
            Block(
                position=Point(x=1 + x, y=5 + y, z=6 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            Block(
                position=Point(x=1 + x, y=5 + y, z=7 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
            # Activate
            Block(
                position=Point(x=1 + x, y=5 + y, z=8 + z),
                type=REDSTONE_BLOCK,
                orientation=NORTH,
            ),
        ]
    )
)


blocks = client.readCube(
    Cube(min=Point(x=x - 10, y=y, z=z - 10), max=Point(x=x + 10, y=y + 10, z=z + 10))
)

for b in blocks.blocks:
    if b.type != 5:
        print(b)
# print(blocks.blocks)
