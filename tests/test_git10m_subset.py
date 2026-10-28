import os

import pyarrow as pa
import pytest

import eotts.data.git10m_subset as gs


def test_shard_layout():
    assert gs.SHARD_STARTS[0] == 0
    assert gs.SHARD_STARTS[-1] == gs.GIT10M_NUM_ROWS
    counts = [gs.shard_num_rows(i) for i in range(gs.GIT10M_NUM_SHARDS)]
    assert sum(counts) == gs.GIT10M_NUM_ROWS
    assert set(counts) == {13346, 13347}
    assert gs.shard_num_rows(308) == 13346          # verified against the Hub file
    assert gs.shard_of(4121463) == 308               # first spatial-split row
    assert gs.shard_of(0) == 0 and gs.shard_of(gs.GIT10M_NUM_ROWS - 1) == 786
    with pytest.raises(IndexError):
        gs.shard_of(gs.GIT10M_NUM_ROWS)


def _fake_shard(shard, path):
    n = gs.shard_num_rows(shard)
    start = gs.SHARD_STARTS[shard]
    t = pa.table({
        "image": pa.array([{"bytes": f"img{start + i}".encode(), "path": None} for i in range(n)]),
        "img_name": [f"tile_{start + i}.jpg" for i in range(n)],
        "caption": [f"caption {start + i}" for i in range(n)],
    })
    with pa.OSFile(path, "wb") as f, pa.ipc.new_stream(f, t.schema) as w:
        w.write_table(t)


def test_build_and_read(tmp_path, monkeypatch):
    rows = [{"hf_idx": i, "img_name": f"tile_{i}.jpg"}
            for i in (gs.SHARD_STARTS[3] + 5, gs.SHARD_STARTS[3] + 7, gs.SHARD_STARTS[10])]

    def fake_download(repo_id, filename, repo_type, revision, local_dir, token=None):
        shard = int(filename.split("-")[1])
        p = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _fake_shard(shard, p)
        return p

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    out = tmp_path / "subset"
    gs.build_subset(rows, str(out))
    assert sorted(os.listdir(out / "parts")) == ["shard-00003.parquet", "shard-00010.parquet"]
    assert not (out / "_shard_tmp").exists()

    sub = gs.Git10MSubset(str(out))
    assert len(sub) == 3
    sub.check_covers(rows)
    r = sub.rows[rows[1]["hf_idx"]]
    assert r.img_name == rows[1]["img_name"] and r.caption == f"caption {rows[1]['hf_idx']}"
    with pytest.raises(KeyError):
        sub[1]


def test_name_mismatch_is_fatal(tmp_path):
    p = tmp_path / "s.arrow"
    _fake_shard(5, str(p))
    table = gs._read_shard(str(p))
    bad = [{"hf_idx": gs.SHARD_STARTS[5] + 1, "img_name": "something_else.jpg"}]
    with pytest.raises(ValueError, match="split metadata has"):
        gs._extract(table, 5, bad)
