from myutils.json import load_json, load_json_or_create, save_json
from tests.common import get_temp_fpath


def test_json():
    temp_fpath = get_temp_fpath("json")

    obj = {"key1": [1, 2, 3], "key2": "string"}

    save_json(obj, temp_fpath)
    loaded_obj = load_json(temp_fpath)

    assert isinstance(loaded_obj, dict)

    assert set(obj) == set(loaded_obj)

    for key, value in obj.items():
        assert value == loaded_obj[key]

    temp_fpath.unlink()
    assert not temp_fpath.exists()

    created_obj = load_json_or_create(temp_fpath, dict)
    assert isinstance(created_obj, dict)
    assert len(created_obj) == 0
