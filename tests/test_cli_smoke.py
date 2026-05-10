from sparse_view_dataset.cli import app


def test_cli_has_subcommands():
    registered = {cmd.name for cmd in app.registered_commands}
    assert "crop" in registered
    assert "project" in registered
