"""Vision module smoke tests (skipped automatically if cv2/mediapipe absent).

Cannot validate face *detection accuracy* without real footage — that's the
Phase 5 hand-labeled-clip evaluation on the target hardware. What we validate
here: the tracker initializes, processes frames of various shapes without
crashing, publishes nothing on faceless frames, and expires stale tracklets.
"""
import asyncio
import time

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("mediapipe")

from aura.bus import EventBus                       # noqa: E402
from aura.perception.attention import AttentionTracker, _iou  # noqa: E402


@pytest.mark.asyncio
async def test_faceless_frames_publish_nothing():
    bus = EventBus()
    seen = []
    bus.subscribe("AttentionEvent", lambda e: _collect(seen, e))
    tracker = AttentionTracker(bus)
    for shape in [(480, 640, 3), (720, 1280, 3), (240, 320, 3)]:
        frame = np.random.randint(0, 40, shape, dtype=np.uint8)  # dark noise
        out = await tracker.process_frame(frame)
        assert out == []
    await bus.drain()
    assert seen == []


async def _collect(acc, e):
    acc.append(e)


def test_iou_geometry():
    a = (0.1, 0.1, 0.5, 0.5)
    assert _iou(a, a) == pytest.approx(1.0)
    assert _iou(a, (0.6, 0.6, 0.9, 0.9)) == 0.0
    assert 0.0 < _iou(a, (0.3, 0.3, 0.7, 0.7)) < 1.0


@pytest.mark.asyncio
async def test_stale_tracklet_expiry():
    bus = EventBus()
    tracker = AttentionTracker(bus)
    from aura.perception.attention import _Tracklet
    old = _Tracklet(person_id="person_x", box=(0, 0, 1, 1))
    old.last_seen = time.time() - 10.0
    tracker._tracklets.append(old)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    await tracker.process_frame(frame)
    assert tracker._tracklets == [], "stale tracklet should be expired"


@pytest.mark.asyncio
async def test_real_face_pose_and_attention(tmp_path):
    """Fetch MediaPipe's own (Apache-2.0) test face and validate the full
    perception path: detection -> stable tracklet -> sane frontal pose ->
    high attention score. Skips offline."""
    import urllib.request
    url = ("https://raw.githubusercontent.com/google-ai-edge/mediapipe/"
           "master/mediapipe/objc/testdata/sergey.png")
    img_path = tmp_path / "face.png"
    try:
        urllib.request.urlretrieve(url, img_path)
    except OSError:
        pytest.skip("no network access for test image")

    bus = EventBus()
    tracker = AttentionTracker(bus)
    frame = cv2.imread(str(img_path))
    events = []
    for _ in range(6):  # let the EMA settle
        events = await tracker.process_frame(frame)
    await bus.drain()

    assert len(events) == 1, "exactly one face expected"
    e = events[0]
    assert abs(e.yaw) < 15 and abs(e.pitch) < 15, f"frontal pose off: {e}"
    assert e.attention > 0.7, f"frontal face should score high: {e.attention}"
    assert e.person_id == "person_0", "tracklet id should be stable"
