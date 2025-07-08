import argparse
from pathlib import Path

from neurosim import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--settings", type=str, default=None)
parser.add_argument("--world_rate", type=int, default=1000)
parser.add_argument("--control_rate", type=int, default=100)
parser.add_argument("--sim_time", type=int, default=20)

parser.add_argument("--save_png", type=str, default=None, help="Save the simulation data in PNG format")
parser.add_argument("--save_h5", type=str, default=None, help="Save the simulation data in HDF5 format")
parser.add_argument("--display", action="store_true", help="Display the simulation data")

args = parser.parse_args()


def main():
    settings = Path(args.settings) if args.settings else None
    sim = Simulator(settings, args.world_rate, args.control_rate, args.sim_time)
    sim.simulate_traj(args.save_h5, args.save_png, args.display)


if __name__ == "__main__":
    main()
