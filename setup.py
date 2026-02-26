import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop


def get_env_python():
    """
    Return the Python executable for the *currently activated* environment.

    sys.executable is unreliable inside a nested pip invocation: it may
    point to the build-isolation venv or the base-conda python rather than
    the user's active conda/virtual-env python.  CONDA_PREFIX and
    VIRTUAL_ENV are set by the shell activation script and are the most
    reliable indicators of where the user actually wants packages installed.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "python")
        if os.path.isfile(candidate):
            return candidate

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = os.path.join(virtual_env, "bin", "python")
        if os.path.isfile(candidate):
            return candidate

    return sys.executable


class CustomInstallCommand(install):
    """Custom installation command that handles dependencies."""

    def run(self):
        print("Running Neurosim custom installation...")
        install.run(self)
        self.run_custom_install()

    def run_custom_install(self):
        """Run the custom installation steps."""
        print("Starting custom dependency installation...")

        if "CONDA_DEFAULT_ENV" not in os.environ and "VIRTUAL_ENV" not in os.environ:
            print("⚠️  WARNING: Not running in a conda/virtual environment!")
            print("   Recommended: conda create -n neurosim python=3.10 cmake=3.14.0")
            print("   Then: conda activate neurosim\n")
        else:
            env_name = os.environ.get("CONDA_DEFAULT_ENV", "virtual environment")
            print(f"✓ Running in environment: {env_name}\n")

        self.install_habitat_sim()
        self.install_pip_requirements()
        self.install_neurosim_cu_esim()

        print("✓ Neurosim installation completed!")

    def install_pip_requirements(self):
        """Install requirements from requirements.txt in current directory."""
        requirements_file = os.path.join(os.getcwd(), "requirements.txt")

        if os.path.exists(requirements_file):
            print("📋 Installing main requirements...")
            python_exe = get_env_python()
            try:
                subprocess.run(
                    [python_exe, "-m", "pip", "install", "-r", requirements_file],
                    check=True,
                )
                print("   ✓ pip requirements installed successfully!\n")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Error installing pip requirements: {e}")
                raise
        else:
            print("   ⚠️  No requirements.txt found in current directory, skipping...\n")

    def install_neurosim_cu_esim(self):
        """Install neurosim_cu_esim from GitHub."""
        print("🚀 Installing neurosim_cu_esim, for CUDA event simulator support...")
        deps_dir = os.path.join(os.getcwd(), "deps")
        esim_dir = os.path.join(deps_dir, "neurosim_cu_esim")

        if not os.path.exists(esim_dir):
            print("   - Cloning neurosim_cu_esim repository...")
            os.makedirs(deps_dir, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/grasp-lyrl/neurosim_cu_esim.git",
                    esim_dir,
                ],
                check=True,
            )
            print("   ✓ Repository cloned successfully")
        else:
            print("   ✓ neurosim_cu_esim repository already exists")

        try:
            print("   - Installing neurosim_cu_esim...")
            python_exe = get_env_python()
            # --no-build-isolation: use the torch already installed by requirements.txt
            # instead of letting pip spin up a throwaway build-venv and download a
            # potentially different torch version.
            subprocess.run(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    esim_dir,
                ],
                check=True,
            )
            print("   ✓ neurosim_cu_esim installation completed successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error installing neurosim_cu_esim: {e}")
            raise

    def install_habitat_sim(self):
        """Install Habitat-Sim with required configuration."""
        print("🏠 Installing Habitat-Sim...")

        deps_dir = os.path.join(os.getcwd(), "deps")
        habitat_dir = os.path.join(deps_dir, "habitat-sim")

        # Clone habitat-sim if not exists
        if not os.path.exists(habitat_dir):
            print("   - Cloning Habitat-Sim repository...")
            os.makedirs(deps_dir, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--branch",
                    "v0.3.3",
                    "https://github.com/facebookresearch/habitat-sim.git",
                    habitat_dir,
                ],
                check=True,
            )
            print("   ✓ Repository cloned successfully")

            ################ change numpy version in habitat-sim requirements.txt ################
            requirements_file = os.path.join(habitat_dir, "requirements.txt")
            with open(requirements_file, "r") as f:
                requirements = [
                    line if not line.startswith("numpy==") else "numpy>=1.26.4\n"
                    for line in f.readlines()
                ]
            with open(requirements_file, "w") as f:
                f.writelines(requirements)
            print("   ✓ Updated numpy version in Habitat-Sim requirements.txt")
            ######################################################################################
        else:
            print("   ✓ Habitat-Sim repository already exists")

        # Install habitat-sim
        original_dir = os.getcwd()
        try:
            os.chdir(habitat_dir)
            python_exe = get_env_python()

            print("   - Installing Habitat-Sim Python requirements...")
            subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True,
            )
            print("   ✓ Requirements installed")

            print("   - Building and installing Habitat-Sim ...")
            print("     Please be patient, this step compiles C++ code...")

            process = subprocess.Popen(
                [
                    python_exe,
                    "setup.py",
                    "install",
                    "--with-cuda",
                    "--headless",
                    "--bullet",
                    "--lto",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            for line in process.stdout:
                print(f"     {line.strip()}")

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, "habitat-sim build"
                )

            print("   ✓ Habitat-Sim installation completed successfully!\n")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error installing Habitat-Sim: {e}")
            raise
        finally:
            os.chdir(original_dir)


class CustomDevelopCommand(develop):
    """Custom develop command that handles dependencies."""

    def run(self):
        print("Running Neurosim development installation...")
        develop.run(self)
        installer = CustomInstallCommand(self.distribution)
        installer.run_custom_install()


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neurosim",
    version="0.2.0",
    author="grasp-lyrl",
    packages=["neurosim"],
    package_dir={"": "src"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        # Add any direct Python dependencies here
    ],
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
    },
    description="A habitat sim based full functional quadrotor simulator with event camera support. real. time.",
)
