from llm_chain_sidecar.models.registry import License, ModelEntry, ModelRegistry


def test_registry_loads_allowlist():
    reg = ModelRegistry.load_default()
    assert len(reg.entries()) > 5
    assert all(isinstance(e, ModelEntry) for e in reg.entries())


def test_default_excludes_restricted_licenses():
    reg = ModelRegistry.load_default()
    assert all(e.license in (License.APACHE_2_0, License.MIT) for e in reg.entries())
    assert all(not e.restricted for e in reg.entries())


def test_include_restricted_surfaces_llama_gemma_deepseek():
    reg = ModelRegistry.load_default()
    restricted = [e for e in reg.entries(include_restricted=True) if e.restricted]
    families = {e.family for e in restricted}
    assert {"Llama", "Gemma", "DeepSeek"}.issubset(families)
    assert all(e.license_caveat for e in restricted)


def test_filter_by_max_params():
    reg = ModelRegistry.load_default()
    small = reg.fitting_within(500_000_000)
    assert all(e.params <= 500_000_000 for e in small)
    assert any("Pythia" in e.name or "SmolLM" in e.name for e in small)


def test_fitting_within_excludes_restricted_by_default():
    reg = ModelRegistry.load_default()
    fits = reg.fitting_within(20_000_000_000)
    assert all(not e.restricted for e in fits)
    fits_with = reg.fitting_within(20_000_000_000, include_restricted=True)
    assert any(e.restricted for e in fits_with)
