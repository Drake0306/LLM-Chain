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


def test_qwen2_vl_entries_present_with_vision_modalities():
    reg = ModelRegistry.load_default()
    by_id = {e.id: e for e in reg.entries()}
    assert "Qwen/Qwen2-VL-2B-Instruct" in by_id
    vlm = by_id["Qwen/Qwen2-VL-2B-Instruct"]
    assert "image" in vlm.modalities
    assert "text" in vlm.modalities


def test_required_modalities_filters_to_multimodal_only():
    reg = ModelRegistry.load_default()
    vlms = reg.entries(required_modalities=["image"])
    assert vlms
    assert all("image" in e.modalities for e in vlms)
    # Should be a strict subset — text-only entries must not appear.
    assert all(e.family == "Qwen2-VL" for e in vlms)


def test_required_modalities_empty_list_is_treated_as_no_filter():
    reg = ModelRegistry.load_default()
    full = reg.entries()
    assert reg.entries(required_modalities=[]) == full


def test_fitting_within_supports_required_modalities():
    reg = ModelRegistry.load_default()
    small_vlm = reg.fitting_within(3_000_000_000, required_modalities=["image"])
    assert all("image" in e.modalities for e in small_vlm)
    assert all(e.params <= 3_000_000_000 for e in small_vlm)
