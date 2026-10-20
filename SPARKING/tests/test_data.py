import hashlib
import json

from sparking.data import OFFICIAL_SPLIT_CANONICAL_SHA256, official_folds, official_split_manifest


def test_official_splits_are_complete_disjoint_and_hash_bound():
    folds = official_folds()
    assert [tuple(map(len, fold)) for fold in folds] == [
        (3683, 921), (3683, 921), (3683, 921), (3683, 921), (3684, 920)
    ]
    for train, test in folds:
        assert not set(train) & set(test)
        assert len(set(train) | set(test)) == 4604
    payload = json.dumps(official_split_manifest(), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == OFFICIAL_SPLIT_CANONICAL_SHA256

