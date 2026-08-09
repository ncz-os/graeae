from __future__ import annotations

import importlib


def test_graeae_package_imports_with_post_nats_core():
    module = importlib.import_module("mnemos.domain.graeae")

    assert module.GraeaeEngine is not None
