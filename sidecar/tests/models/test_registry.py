from llm_chain_sidecar.models.registry import License, ModelEntry, ModelRegistry


def test_registry_loads_allowlist():
    reg = ModelRegistry.load_default()
    assert len(reg.entries) > 5
    assert all(isinstance(e, ModelEntry) for e in reg.entries)


def test_default_excludes_restricted_licenses():
    reg = ModelRegistry.load_default()
    assert all(e.license in (License.APACHE_2_0, License.MIT) for e in reg.entries)


def test_filter_by_max_params():
    reg = ModelRegistry.load_default()
    small = reg.fitting_within(500_000_000)
    assert all(e.params <= 500_000_000 for e in small)
    assert any("Pythia" in e.name or "SmolLM" in e.name for e in small)
