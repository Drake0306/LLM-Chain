import queue

import tqdm.auto

from llm_chain_sidecar.trainers.hf_progress import emit_hf_download_progress


def _drain(q: queue.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_emit_hf_download_progress_pushes_event_on_each_advance():
    q: queue.Queue = queue.Queue()
    with emit_hf_download_progress(q):
        bar = tqdm.auto.tqdm(total=100, desc="model.safetensors", mininterval=0)
        bar.n = 50
        bar.refresh()
        bar.n = 100
        bar.refresh()
        bar.close()
    events = _drain(q)
    descs = {e["desc"] for e in events}
    bytes_done = {e["bytes_done"] for e in events}
    bytes_total = {e["bytes_total"] for e in events}
    assert descs == {"model.safetensors"}
    assert 50 in bytes_done
    assert 100 in bytes_done
    assert bytes_total == {100}


def test_emit_hf_download_progress_throttles_redundant_emits():
    """A 1 GB download fires thousands of tqdm.update() calls. We must not
    emit one event per call or we'll choke the SSE channel."""
    q: queue.Queue = queue.Queue()
    with emit_hf_download_progress(q):
        bar = tqdm.auto.tqdm(total=10000, desc="big.bin", mininterval=0)
        for n in range(0, 10001, 1):
            bar.n = n
            # display() is what we patch; call it directly to bypass tqdm's
            # own refresh-rate throttle and exercise just our throttle.
            bar.display()
        bar.close()
    events = _drain(q)
    # 100 buckets at 1% granularity, plus the final n==total tick that always
    # fires regardless. A few extra is fine; thousands is not.
    assert 50 < len(events) < 200


def test_emit_hf_download_progress_skips_indeterminate_bars():
    q: queue.Queue = queue.Queue()
    with emit_hf_download_progress(q):
        bar = tqdm.auto.tqdm(total=None, desc="streaming", mininterval=0)
        bar.n = 5
        bar.refresh()
        bar.close()
    assert _drain(q) == []


def test_emit_hf_download_progress_restores_display_method():
    original = tqdm.auto.tqdm.display
    q: queue.Queue = queue.Queue()
    with emit_hf_download_progress(q):
        assert tqdm.auto.tqdm.display is not original
    assert tqdm.auto.tqdm.display is original


def test_emit_hf_download_progress_restores_on_exception():
    original = tqdm.auto.tqdm.display
    q: queue.Queue = queue.Queue()
    try:
        with emit_hf_download_progress(q):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tqdm.auto.tqdm.display is original
