import argparse
import logging
from pathlib import Path

from neurosim import SynchronousSimulator

parser = argparse.ArgumentParser(description="Run Neurosim simulation")
parser.add_argument(
    "--settings", type=str, required=True, help="Path to settings YAML file"
)
parser.add_argument(
    "--display", action="store_true", help="Display live visualization with Rerun"
)
parser.add_argument(
    "--log-h5", type=str, default=None, help="Path to HDF5 file for logging data"
)
parser.add_argument(
    "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging"
)
args = parser.parse_args()


def main():
    # Setup logging level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )

    settings_path = Path(args.settings)
    sim = SynchronousSimulator(settings_path)
    sim.run(display=args.display, log_h5=args.log_h5)


if __name__ == "__main__":
    main()
