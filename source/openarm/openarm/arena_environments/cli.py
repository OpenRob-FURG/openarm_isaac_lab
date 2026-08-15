from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from openarm.arena_environments.assembly_nut_and_bolt import BimanualOpenArmAssemblyNutAndBoltEnvironment
from openarm.arena_environments.open_microwave import BimanualOpenArmOpenMicrowaveEnvironment
from openarm.arena_environments.pick_and_place_exhaust_pipe import BimanualOpenArmPickAndPlaceExhaustPipeEnvironment
from openarm.arena_environments.pick_exhaust_pipe import BimanualOpenArmPickExhaustPipeEnvironment

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder


# Collection of the available openarm environments
ExampleEnvironments = {
    BimanualOpenArmAssemblyNutAndBoltEnvironment.name: BimanualOpenArmAssemblyNutAndBoltEnvironment,
    BimanualOpenArmOpenMicrowaveEnvironment.name: BimanualOpenArmOpenMicrowaveEnvironment,
    BimanualOpenArmPickAndPlaceExhaustPipeEnvironment.name: BimanualOpenArmPickAndPlaceExhaustPipeEnvironment,
    BimanualOpenArmPickExhaustPipeEnvironment.name: BimanualOpenArmPickExhaustPipeEnvironment,
}


def parse_and_return_external_environment_from_string(environment_path: str) -> dict[str, Any]:
    """Parse a string and import the environment class

    Args:
        environment_path: The path to the environment class

    Raises:
        ValueError: If the environment path is not in the format "module_path:class_name"

    Returns:
        dict[str, Any]: A dictionary with the environment name as the key and the environment class as the value
    """
    # Parse the environment path and import the environment class
    # We assume the environment path is in the format "module_path:class_name"
    # Add a check for the format
    if ":" not in environment_path:
        raise ValueError(f"Invalid environment path: {environment_path}. Expected format: 'module_path:class_name'")
    module_path, class_name = environment_path.split(":", 1)
    try:
        module = importlib.import_module(module_path)
        environment_class = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Could not resolve the environment path '{environment_path}' into an environment class."
            " The format should be 'module_path:class_name'.\n"
            f"Received the error:\n {e}."
        ) from e
    name = getattr(environment_class, "name", environment_class.__name__)
    assert name is not None, "Environment class must have a 'name' attribute"
    return {name: environment_class}


def add_example_environments_cli_args(args_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Parse the parser once here to add the external environments to the example environments
    args, unknown = args_parser.parse_known_args()
    environment = getattr(args, "external_environment_class_path", None)
    if environment is not None:
        # Update the ExampleEnvironments dictionary with the new external environment
        print(f"Adding external environment: {environment}")
        ExampleEnvironments.update(parse_and_return_external_environment_from_string(environment))
    subparsers = args_parser.add_subparsers(
        dest="example_environment", required=True, help="Example environment to run"
    )
    for example_environment in ExampleEnvironments.values():
        subparser = subparsers.add_parser(example_environment.name)
        example_environment.add_cli_args(subparser)

    return args_parser


def get_isaaclab_arena_environments_cli_parser(
    args_parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if args_parser is None:
        args_parser = get_isaaclab_arena_cli_parser()
    # NOTE: This command adds subparsers for each example environment.
    # So it has to be added last, because the subparser flags are parsed after the others.
    args_parser = add_example_environments_cli_args(args_parser)
    return args_parser


def get_arena_builder_from_cli(args_cli: argparse.Namespace) -> ArenaEnvBuilder:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Get the example environment
    assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    assert (
        args_cli.example_environment in ExampleEnvironments
    ), f"Example environment type {args_cli.example_environment} not supported"
    example_env = ExampleEnvironments[args_cli.example_environment]()

    # Compile the environment
    env_builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    return env_builder
