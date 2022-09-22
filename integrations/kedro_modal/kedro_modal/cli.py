import click
from modal import lookup

from .modal_functions import main_stub, sync_stub


@click.group(name="Kedro-Modal")
def commands():
    """Kedro plugin for running kedro pipelines on Modal"""
    pass


@commands.group(
    name="modal", context_settings=dict(help_option_names=["-h", "--help"])
)
def modal_group():
    """Interact with Kedro pipelines run on Modal"""


@modal_group.command(help="Run kedro project on Modal")
@click.pass_obj
def run(metadata):
    stub, remote_project_mount_path, remote_data_path = main_stub(metadata.project_path, metadata.project_name, metadata.package_name)
    with stub.run() as app:
        app.sync_data(remote_project_mount_path / "data", remote_data_path, reset=False)
        app.run_kedro(remote_project_mount_path, remote_data_path)

@modal_group.command(help="Run kedro project on Modal")
@click.pass_obj
def debug(metadata):
    stub, remote_project_mount_path, remote_data_path = main_stub(metadata.project_path, metadata.project_name, metadata.package_name)
    stub.interactive_shell()

@modal_group.command(help="Deploy kedro project to Modal, scheduling it to run daily")
@click.pass_obj
def deploy(metadata):
    stub, remote_project_mount_point, remote_data_path = main_stub(metadata.project_path, metadata.project_name, metadata.package_name)
    name=f"kedro.{metadata.project_name}"
    stub.deploy(name)
    sync_data = lookup(name, "sync_data")  # use the deployed function
    sync_data(remote_project_mount_point / "data", remote_data_path)


@modal_group.command(short_help="Sync the local data directory to Modal", help="Sync the local data directory to Modal, overwriting any existing data there")
@click.pass_obj
def reset(metadata):
    stub, source_path, destination_path = sync_stub(metadata.project_path, metadata.project_name)
    with stub.run() as app:
        app.sync_data(source_path, destination_path, reset=True)
