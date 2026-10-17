import numpy as np

from soundgen.controls import Controls, apply_controls, onsets
from soundgen.smoke import check_flow, main

FPS = 22050 / 256


def _steps_sketch():
    db = np.full((20, 173), -100.0)
    for o in (5, 48, 91, 134):
        for k in range(12):
            db[:, o + k] = -20 - 3 * k
    return (db + 100) / 100


def test_flow_inverse_and_logdet():
    check_flow()


def test_controls_move_the_right_way():
    s = _steps_sketch()
    level = lambda x: 10 * np.log10(np.mean(10 ** ((x * 100 - 100) / 10)))
    assert level(apply_controls(s, Controls(intensity=1), "sequence")) > level(s) + 6
    assert level(apply_controls(s, Controls(distance=1), "sequence")) < level(s) - 10
    fast = apply_controls(s, Controls(speed=2.0), "sequence")
    gaps = np.diff(onsets(fast * 100 - 100, FPS))
    assert abs(np.median(gaps) - 21.5) < 2
    long = apply_controls(s, Controls(duration=6.0), "texture")
    assert long.shape[1] == int(round(6.0 * FPS)) + 1


def test_end_to_end_smoke():
    main("cpu")
