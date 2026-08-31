"""Assert-based checks for the streaming pipeline. Run:

    python -m tracking.tests.sanity_check_stream
"""
import numpy as np

from tracking.core.blob_tracker import merged_center
from tracking.stream.stream import Smoother, StreamPipeline, SPAN, BUFFER, jitter


def make_moving_bar(T=40, H=120, W=240, speed=3, bar_h=60, bar_w=25):
    """A bright bar drifting right on a flat background - the person stand-in."""
    frames = np.full((T, H, W), 40, dtype=np.uint8)
    for t in range(T):
        x = 20 + speed * t
        frames[t, 30:30 + bar_h, x:x + bar_w] = 220
    return frames


def check_merged_center_box_is_additive():
    dets = [dict(x=10.0, y=10.0, area=100, bbox=(5, 0, 15, 20)),
            dict(x=12.0, y=40.0, area=100, bbox=(8, 30, 16, 50)),
            dict(x=500.0, y=500.0, area=100, bbox=(495, 495, 505, 505))]  # far away, excluded
    cx, cy = merged_center(dets, 11.0, 25.0, merge_radius=40.0)
    cx2, cy2, box = merged_center(dets, 11.0, 25.0, merge_radius=40.0, return_box=True)
    assert (cx, cy) == (cx2, cy2), f"return_box changed the center: {(cx, cy)} vs {(cx2, cy2)}"
    assert box == (5, 0, 16, 50), f"expected the merged box (5, 0, 16, 50), got {box}"
    assert ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) == (cx, cy), "center is not the box center"
    assert merged_center([], 1.0, 2.0, 40.0, return_box=True) == (1.0, 2.0, None)
    print("merged_center ok: box returned, center unchanged")


def check_pipeline_emits_for_the_buffer_center():
    frames = make_moving_bar()
    # suppress_warmup=False: this checks ring-buffer geometry, not MOG2's adaptation, and the
    # synthetic clip is shorter than the default bg_frames warm-up
    pipe = StreamPipeline(person_height=60.0, readout='center', suppress_warmup=False)
    results = [r for r in (pipe.step(f) for f in frames) if r is not None]
    assert SPAN == 6 and BUFFER == 13, f"geometry changed: SPAN={SPAN}, BUFFER={BUFFER}"
    assert len(results) == len(frames) - (BUFFER - 1), \
        f"expected {len(frames) - (BUFFER - 1)} emissions, got {len(results)}"
    assert results[0].frame_index == SPAN, \
        f"first emission should describe frame {SPAN}, got {results[0].frame_index}"
    assert all(np.array_equal(r.frame, frames[r.frame_index]) for r in results), \
        "Result.frame is not the frame it claims to describe"
    xs = [r.x for r in results if r.x is not None]
    assert len(xs) > len(results) // 2, f"only {len(xs)}/{len(results)} results tracked anything"
    assert xs == sorted(xs), "tracked x should increase monotonically for a rightward bar"
    boxed = [r for r in results if r.box is not None]
    assert len(boxed) > len(results) // 2, f"only {len(boxed)}/{len(results)} results carry a box"
    print(f"pipeline ok: {len(results)} emissions, {len(boxed)} with boxes, x monotonic")


def check_newest_readout_is_zero_latency():
    frames = make_moving_bar()
    pipe = StreamPipeline(person_height=60.0, readout='newest', suppress_warmup=False)
    results = [r for r in (pipe.step(f) for f in frames) if r is not None]
    assert results[0].frame_index == BUFFER - 1, \
        f"newest readout should describe frame {BUFFER - 1} first, got {results[0].frame_index}"
    assert results[-1].frame_index == len(frames) - 1, \
        f"newest readout should reach the last frame, got {results[-1].frame_index}"
    print(f"newest readout ok: describes frames {results[0].frame_index}..{results[-1].frame_index}")


def check_jitter_is_zero_on_constant_velocity():
    assert abs(jitter(dict(enumerate([0.0, 3.0, 6.0, 9.0, 12.0])))) < 1e-9, "constant velocity must give zero jitter"
    assert jitter(dict(enumerate([0.0, 0.0, 10.0, 0.0, 0.0]))) > 1.0, "a spike must register as jitter"
    print("jitter metric ok")


def check_streaming_matches_the_offline_evaluator(video='data/ido_walk.mkv', scale=0.25,
                                                  person_height=210.0, limit=180):
    """The streaming path must be IDENTICAL to track_windows_in_sequence, not merely close:
    both run one continuous MOG2 pass in chronological order with the same history, the same
    morphology and the same window geometry, so any difference is a bug in the ring buffer's
    bookkeeping rather than an approximation. This is a stronger and cheaper gate than
    comparing two streaming engines against each other."""
    import os
    if not os.path.exists(video):
        print(f"skip offline-parity check: {video} not present")
        return
    from tracking.core.track_sequence import track_windows_in_sequence
    from tracking.stream.stream import frames_from_video, NTH_FRAME

    frames = np.stack([f for _, f in zip(range(limit), frames_from_video(video, scale))])
    stream = {}
    # track_windows_in_sequence has no warm-up suppression, so parity requires it off here
    pipe = StreamPipeline(person_height=person_height, readout='center', suppress_warmup=False)
    for f in frames:
        r = pipe.step(f)
        if r is not None:
            stream[r.frame_index] = r
    centers = sorted(stream)
    # expected_height must be non-None for track_windows_in_sequence to keep the derived
    # shape term that scale_relative_params (and therefore StreamPipeline) always uses
    offline = track_windows_in_sequence(frames, centers, span=SPAN, nth_frame=NTH_FRAME,
                                        person_height=person_height, expected_height=1.0)
    assert len(centers) == len(frames) - (BUFFER - 1), f"emitted {len(centers)} windows"
    worst, n_both = 0.0, 0
    for c in centers:
        a, b = stream[c], offline[c]
        assert (a.x is None) == (b is None), f"frame {c}: streaming {a.x} vs offline {b}"
        if b is None:
            continue
        n_both += 1
        worst = max(worst, abs(a.x - b['x']), abs(a.y - b['y']))
    assert worst == 0.0, f"streaming and offline disagree by up to {worst:.6f}px"
    print(f"offline parity ok: {len(centers)} windows, {n_both} tracked, identical to the bit")


def check_smoother_has_no_lag_on_constant_velocity():
    """Holt's trend term exists precisely so a constant-velocity target is not lagged. A plain
    first-order EMA would sit v*tau behind; this must not."""
    sm = Smoother(person_height=100.0, fps=25.0, halflife_s=0.15)
    v = 5.0
    for t in range(80):
        out, _, coast = sm.update((10.0 + v * t, 50.0), (30.0, 90.0))
    lag = (10.0 + v * 79) - out[0]
    assert abs(lag) < 0.5, f"constant-velocity lag {lag:.2f}px - the trend term is not working"
    print(f"smoother ok: constant-velocity lag {lag:+.3f}px (a plain EMA would lag ~{v * 5:.0f}px)")


def check_smoother_gates_an_outlier_but_relocks_on_a_sustained_jump():
    sm = Smoother(person_height=100.0, fps=25.0, halflife_s=0.15, jump_max=0.75, hold_s=0.35)
    for t in range(40):
        out, _, _ = sm.update((100.0, 50.0), (30.0, 90.0))
    settled = out[0]
    out, _, coast = sm.update((900.0, 50.0), (30.0, 90.0))        # one wild frame
    assert coast, "a 800px jump on a 100px person should be gated, not followed"
    assert abs(out[0] - settled) < 5.0, f"gated frame still moved the estimate to {out[0]:.0f}"
    for t in range(40):                                            # sustained: must re-lock
        out, _, coast = sm.update((900.0, 50.0), (30.0, 90.0))
    assert abs(out[0] - 900.0) < 5.0, f"never re-locked; stuck at {out[0]:.0f}"
    print("smoother ok: single outlier gated, sustained jump re-locked")


def check_smoother_constants_are_frame_rate_independent():
    """Half-lives are in seconds, so the same wall-clock smoothing must result at any fps."""
    outs = []
    for fps in (12.0, 24.0, 48.0):
        sm = Smoother(person_height=100.0, fps=fps, halflife_s=0.2)
        n = int(round(2.0 * fps))                                  # two seconds either way
        for t in range(n):
            out, _, _ = sm.update((0.0 if t < n // 2 else 60.0, 0.0), (30.0, 90.0))
        outs.append(out[0])
    assert max(outs) - min(outs) < 2.0, f"fps-dependent smoothing: {outs}"
    print(f"smoother ok: frame-rate independent ({['%.2f' % o for o in outs]} at 12/24/48 fps)")


def check_a_frozen_window_yields_no_box():
    """A window of duplicate frames must produce NO estimate, not a zero-score one. Nothing in
    the code special-cases this: MOG2 absorbs a static frame into its background model, so
    duplicates stop producing foreground, no track reaches min_track_length, and score_and_fit
    returns None. Every parameter of this tracker is defined on motion, so with no motion there
    is nothing to report - which is the honest behaviour and worth pinning."""
    moving = make_moving_bar(T=40)
    frozen = np.concatenate([moving, np.repeat(moving[-1:], 3 * BUFFER, axis=0)])
    pipe = StreamPipeline(person_height=60.0, suppress_warmup=False)
    results = [r for r in (pipe.step(f) for f in frozen) if r is not None]
    inside = [r for r in results if r.frame_index >= len(moving) + SPAN]
    assert inside, "no emissions landed fully inside the frozen block"
    boxed = [r.frame_index for r in inside if r.box is not None]
    assert not boxed, f"frames {boxed} still produced a box inside a fully frozen window"
    assert all(r.x is None for r in inside), "a position was reported with no motion at all"
    print(f"frozen window ok: {len(inside)} emissions fully inside the freeze, none with a box")


def main():
    check_merged_center_box_is_additive()
    check_a_frozen_window_yields_no_box()
    check_smoother_has_no_lag_on_constant_velocity()
    check_smoother_gates_an_outlier_but_relocks_on_a_sustained_jump()
    check_smoother_constants_are_frame_rate_independent()
    check_jitter_is_zero_on_constant_velocity()
    check_pipeline_emits_for_the_buffer_center()
    check_newest_readout_is_zero_latency()
    check_streaming_matches_the_offline_evaluator()


if __name__ == '__main__':
    main()
