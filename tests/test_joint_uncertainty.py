import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")
from model_engine.ogs2_joint_geometry import close_unknown_boundary


def masks():
    hard=np.zeros((30,40),bool)
    hard[10:16,10:16]=True
    static=np.ones(hard.shape,bool)
    valid=np.ones(hard.shape,bool)
    return hard,static,valid


def test_whole_uncertain_penumbra_stays_pure_source():
    hard,static,valid=masks()
    static[12:14,15:25]=False
    result,info=close_unknown_boundary(hard,static,valid,valid)
    assert result[12:14,15:25].all()
    assert info["addedUncertainPixels"]==18
    assert not np.array_equal(result,hard)
    assert hard.sum()==36


def test_unknown_component_cannot_be_clipped_at_unseen_pixels():
    hard,static,valid=masks()
    static[12:14,15:25]=False
    valid[12:14,24]=False
    with pytest.raises(ValueError,match="unseen"):
        close_unknown_boundary(hard,static,valid,np.ones_like(valid))


def test_remote_unknowns_are_not_written():
    hard,static,valid=masks()
    static[12:14,30:35]=False
    result,info=close_unknown_boundary(hard,static,valid,valid)
    np.testing.assert_array_equal(result,hard)
    assert info["addedUncertainPixels"]==0


def test_large_uncertain_transition_fails_atomically():
    hard,static,valid=masks()
    static[12:14,15:25]=False
    with pytest.raises(ValueError,match="budget"):
        close_unknown_boundary(hard,static,valid,valid,max_pixels=40)
    assert hard.sum()==36
