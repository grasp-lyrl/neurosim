"""Tiny cortex state publisher used by the bridge smoke test.

Publishes DictMessage `state` frames at 10 Hz forever; the test orchestrator
kills the process.
"""

import cortex
from cortex.core.node import Node
from cortex.messages.standard import DictMessage


class Publisher(Node):
    def __init__(self):
        super().__init__("smoke_state_pub")
        self.pub = self.create_publisher("state", DictMessage, queue_size=10)
        self.count = 0
        self.create_timer(0.1, self.tick)

    async def tick(self) -> None:
        self.count += 1
        msg = {
            "x": [float(self.count), 0.0, 1.0],
            "q": [0.0, 0.0, 0.0, 1.0],
            "v": [0.1, 0.2, 0.3],
            "w": [0.01, 0.02, 0.03],
            "timestamp": self.count * 0.1,
            "simsteps": self.count,
        }
        ok = self.pub.publish(DictMessage(data=msg))
        if self.count % 10 == 0:
            print(f"[pub] tick {self.count} ok={ok}", flush=True)


async def main():
    async with Publisher() as node:
        await node.run()


if __name__ == "__main__":
    cortex.run(main())
