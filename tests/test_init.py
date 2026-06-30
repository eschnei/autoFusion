"""`autofusion init` scaffolds a valid config and won't clobber it (MAR-18)."""

import argparse

from autofusion.cli import _cmd_init
from autofusion.config import DEFAULT_CONFIG_NAME, load_config


def _args(directory, force=False):
    return argparse.Namespace(config=str(directory / DEFAULT_CONFIG_NAME), force=force)


def test_init_writes_loadable_config(tmp_path):
    rc = _cmd_init(_args(tmp_path))
    target = tmp_path / DEFAULT_CONFIG_NAME
    assert rc == 0 and target.exists()
    cfg = load_config(target)  # parses without error
    assert "llama3.2" in cfg.models and cfg.fusion.aggregator == "llama3.2"


def test_init_does_not_overwrite_without_force(tmp_path):
    target = tmp_path / DEFAULT_CONFIG_NAME
    target.write_text("# my hand-edited config\n")
    _cmd_init(_args(tmp_path, force=False))
    assert target.read_text() == "# my hand-edited config\n"  # untouched


def test_init_force_overwrites(tmp_path):
    target = tmp_path / DEFAULT_CONFIG_NAME
    target.write_text("# stale\n")
    _cmd_init(_args(tmp_path, force=True))
    assert "[fusion]" in target.read_text()  # replaced with the starter
