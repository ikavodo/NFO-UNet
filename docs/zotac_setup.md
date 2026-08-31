# Setting up the live person tracker on the Zotac

A runbook for installing `nfo-tracker` as a service that survives reboots, camera unplugs and
weeks of uptime. Follow it in order; step 4 is the one people skip and it is the one that makes
the service fail while the same command works by hand.

Every number and every failure mode below was measured on the development machine
(`L3DSS2520`, Logitech HD Pro Webcam C920). The Zotac will differ in at least the install paths
and the camera's serial number.

**What you get:** a headless process reading the camera at 30 fps, tracking a walking person
under fragmented occlusion, at ~2.5 ms/frame of tracker cost and ~185 MB resident. No GPU, no
model download, no network.

---

## 0. Prerequisites

```bash
python3 --version          # must be >= 3.10 (the code uses `X | None` annotations)
sudo apt install v4l-utils pipx
```

`v4l-utils` gives `v4l2-ctl`, which steps 2 and 3 need. Nothing else is required — the tracker
depends on `numpy`, `opencv-python` and `scipy` only. It does **not** need torch or any of the
U-Net training stack, even though they appear in the repository's `requirements.txt`.

---

## 1. Install the package

```bash
git clone <repo> ~/nfo-unet && cd ~/nfo-unet
pipx install .
```

`pipx` builds an isolated virtualenv and puts the command on `PATH`, so the Zotac needs no conda
and nothing is added to the system Python. Confirm and note the absolute path — the service will
need it:

```bash
which nfo-tracker            # e.g. /home/<user>/.local/bin/nfo-tracker
nfo-tracker --help
```

<details>
<summary>Alternative: <code>pip install</code> into an existing environment</summary>

```bash
pip install .                        # or: pip install -e . --no-deps
pip show -f nfo-tracker | grep Location
```

Use `--no-deps` if the environment already has a working `cv2`/`numpy`/`scipy` you do not want
pip to touch. The package deliberately installs only the `tracking*` packages: the repository
root also holds `config/`, `dataset/`, `eval/`, `network/` and `utils/`, and installing those as
top-level modules would shadow unrelated packages in the same environment.
</details>

---

## 2. Find the camera's capture node

```bash
v4l2-ctl --list-devices
```

```
HD Pro Webcam C920 (usb-0000:80:14.0-2):
        /dev/video2
        /dev/video3
```

**A UVC camera exposes two nodes and only the first streams.** The second is a metadata node.
Confirm which is which — the capture node reports `Video Capture` in its caps:

```bash
v4l2-ctl -d /dev/video2 --all | grep -A1 'Device Caps'
```

Now get a **stable** path, because `/dev/videoN` renumbers across reboots and replugs. On the
development machine the C920 came up as `video2` while the built-in camera held `video0`/`video1`;
a reboot can swap them.

```bash
ls -l /dev/v4l/by-id/ /dev/v4l/by-path/
```

```
usb-046d_HD_Pro_Webcam_C920_363EAC1F-video-index0 -> ../../video2   <-- capture
usb-046d_HD_Pro_Webcam_C920_363EAC1F-video-index1 -> ../../video3   <-- metadata
```

Pick one and keep it:

| path style | follows | use when |
|---|---|---|
| `by-id` | **this camera**, any USB port (contains its serial) | normal case |
| `by-path` | **this USB port**, any camera | the camera may be swapped for an identical one |

Note `-video-index0` is the capture node. `-index1` will not stream.

---

## 3. Check the camera offers MJPG at 30 fps

```bash
v4l2-ctl -d /dev/video2 --list-formats-ext | grep -E "\[[0-9]\]|1920x1080|1280x720" | head
```

This is not a formality. The C920's table:

| format | 1280×720 | 1920×1080 | max resolution |
|---|---|---|---|
| **MJPG** | 30 fps | 30 fps | 1920×1080 @30 |
| YUYV | 10 fps | **5 fps** | 2304×1536 @**2 fps** |

`nfo-tracker` requests MJPG by default and prints what it actually got. If your camera has no
MJPG mode, drop to `--cam-size 640x480`, which is usually 30 fps even in YUYV.

---

## 4. Add the user to the `video` group — MANDATORY

**This is the step that makes the service fail while the identical command works in your
terminal.** `/dev/video*` is `root:video 0660` plus an ACL, and systemd-logind grants the
**active session** user access through that ACL. On the development machine:

```bash
$ getfacl /dev/video2 | grep akovi
user:akovi:rw-              # granted by logind, because there is a login session
$ id -nG akovi | tr ' ' '\n' | grep -c '^video$'
0                           # NOT in the video group at all
```

A service has no session, so it gets no ACL entry and the open fails with permission denied. A
lingering `systemctl --user` service has no session either, so it needs the group too.

```bash
sudo usermod -aG video $USER
```

**Log out and back in** (or reboot) — a new group does not appear in an existing session. Verify:

```bash
id -nG | tr ' ' '\n' | grep '^video$'      # must print: video
```

---

## 5. Run it by hand before touching systemd

Always do this first. It separates a path/permission problem from a service problem.

```bash
nfo-tracker --camera /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_363EAC1F-video-index0 \
            --no-display --status-every 10 --max-frames 900
```

Stay out of frame for ~2 s so the background model forms, then walk across. Healthy output:

```
auto-exposure and auto-WB disabled, static camera assumed
camera '/dev/v4l/by-id/...-video-index0': backend V4L2, negotiated 1280x720 @ 30fps, fourcc MJPG
bootstrapped person height 276px over 120 frames, measured 29.3 fps
[  0.00h] 383 frames   30.0 fps  rss  183.7 MB
[  0.01h] 684 frames   30.0 fps  rss  183.7 MB
```

Check all four:

1. `backend V4L2` — not FFMPEG.
2. `fourcc MJPG` and the fps you expect.
3. A `bootstrapped person height` line appears. It should be a plausible fraction of the
   processing frame height (`--scale 0.5` of 720p = 360 px, so 200–300 px is right for someone a
   few metres away).
4. `rss` is **flat** between status lines.

Drop `--no-display` if the Zotac has a screen and you want to watch it. Thin green = raw
per-frame merged blob box, thick cyan = smoothed, yellow + `COASTING` = the gate rejected a frame
and the filter is running on its own motion model. `ESC` or `q` quits.

---

## 6. Install the service

```bash
sudo cp deploy/nfo-tracker.service /etc/systemd/system/
sudo mkdir -p /etc/nfo-tracker && sudo cp deploy/nfo-tracker.env /etc/nfo-tracker/env
sudo editor /etc/nfo-tracker/env                       # CAMERA, and CAM_SIZE/SCALE if wanted
sudo editor /etc/systemd/system/nfo-tracker.service    # User=, Group=, ExecStart= path
sudo systemctl daemon-reload
sudo systemctl enable --now nfo-tracker
```

Three edits in the unit, all near the top of `[Service]`:

- `User=` / `Group=` — the account you added to `video`
- `ExecStart=` — the absolute path from step 1. **It must be a literal**; systemd expands
  `${VAR}` in ExecStart *arguments* but not in the command name.

Everything else (camera, resolution, scale, status interval) lives in
`/etc/nfo-tracker/env` and needs no unit edit.

---

## 7. Verify

```bash
systemctl status nfo-tracker
journalctl -u nfo-tracker -f
```

You want the same four checks as step 5. Then confirm it actually survives things:

```bash
sudo systemctl restart nfo-tracker              # comes back
# unplug the camera, wait 10 s, plug it back in:
journalctl -u nfo-tracker -n 20                 # expect "camera read failed; reconnecting in 2s"
sudo reboot                                     # comes back after boot
```

The reboot test matters because at boot the USB camera is often not enumerated yet. The first
open fails **by design** (a mistyped device should fail loudly rather than retry forever), and
systemd restarts until the camera appears. The unit sets `StartLimitIntervalSec=0` precisely so
systemd's default "give up after 5 rapid restarts" cannot permanently stop it.

---

## Troubleshooting

Every row here is a failure that actually occurred during development.

| Symptom | Cause | Fix |
|---|---|---|
| `cannot open camera '/dev/videoN'` | pointed at the **metadata** node | use `-video-index0`, not `-index1` |
| Works by hand, service says permission denied | not in `video` group; terminal worked via the logind ACL | step 4, then log out and back in |
| `backend FFMPEG`, fourcc `\x00`, 5–10 fps | a **path** opened without `CAP_V4L2` routes to FFMPEG, which ignores every V4L2 property | `nfo-tracker` forces `CAP_V4L2`; if you wrote your own script, pass it. Measured: 8.9 fps via FFMPEG vs 24.1 via V4L2 |
| 5 fps at 1080p | YUYV negotiated | check `--list-formats-ext`; use MJPG or a smaller size |
| Camera works, then `/dev/videoN` is wrong after a reboot | kernel renumbering | use the `by-id` or `by-path` symlink |
| `BOOTSTRAP` overlay never locks; `no motion detected` | the height needs someone moving in frame | walk across; or `--person-height N` to skip |
| Box is person-sized but tracking is poor on an empty room start | *fixed* — the estimator used to lock onto MOG2's all-foreground first frame (~80% of frame height, and deterministic, so it looked "stable") | nothing to do; the motion gate prevents it |
| Box jumps whenever the lights change | camera auto-gain; a global brightness shift reads as everything-is-foreground | `--auto-exposure 1` (0.25 is the V4L2 manual value, but drivers disagree) |
| Box drifts constantly, nothing ever locks | camera is not rigidly mounted | fix the mount; a static camera is a hard requirement |
| Service stops after a few failures | systemd's default start limit | the shipped unit sets `StartLimitIntervalSec=0`; check you did not remove it |
| Disk fills | `--record` was enabled | it grows ~1 GB/hour at 640×360. Off by default; leave it off |
| Journal grows | 288 status lines/day at `STATUS_EVERY=300` | journald rotates by default; cap with `SystemMaxUse=` in `journald.conf` if needed |

---

## Tuning

Two knobs, both in physical units rather than per-frame, so they behave the same at any frame
rate:

| Problem | Knob | Direction |
|---|---|---|
| Box lags the person | `--halflife` (default 0.15 s) | smaller |
| Box jitters | `--halflife` | larger |
| Box jumps to background objects | `--jump-max` (default 0.75 person-heights) | smaller |
| Box takes too long to re-acquire | `--jump-max` | larger |

Resolution: `--cam-size 1920x1080 --scale 0.5` gives 960×540 processing. The tracker measures
19 ms/frame end-to-end at full 1080p, so there is headroom; 720p with `--scale 0.5` (640×360) is
the tested default.

---

## Known limits

- **Static camera only.** Everything rests on background subtraction. A moving or vibrating
  camera makes every pixel foreground.
- **6 frames of lookahead**, inherent to the centred window — 250 ms at 24 fps. Smoothing adds
  roughly its half-life on top, so budget ~0.4 s end to end.
- **No presence gate.** The tracker reports its best candidate even with nobody in frame; scores
  collapse (median 2 versus 21 with a person) but do not separate cleanly. A frozen video is the
  exception and does correctly produce no box at all.
- **Uptime evidence is tens of minutes, not weeks.** RSS was flat over a 60 000-frame soak
  (353.0 → 353.5 MB at 960×540) and every buffer is a `deque` with `maxlen`, but no week-long run
  has been done. `MemoryMax=1G` in the unit is the backstop.
- **One uncovered failure mode:** the process wedging while still producing frames. Crashes,
  camera loss and camera stalls are all recovered. A true watchdog would need `sd_notify`. The
  manual substitute is the journal: if the status line stops advancing, it is wedged.

---

## Accuracy, for reference

Ground-truth-scored on the four NFO sequences (`data/nfo_processed`, person ~52 px in a 224×224
frame), with the height bootstrapped rather than supplied:

| sequence | hit@0.1 raw | hit@0.1 smoothed | mean residual smoothed |
|---|---|---|---|
| seq1 | 0.986 | 1.000 | 0.021 |
| seq2 | 0.967 | 0.998 | 0.018 |
| seq3 | 0.983 | 1.000 | 0.027 |
| seq4 | 0.988 | 0.988 | 0.027 |

`hit@0.1` is the fraction of frames whose reported centre is within 10% of the frame diagonal of
the ground-truth box centre. Those numbers come from 800×600-derived NFO footage; they do not
transfer unchanged to your room, where there is no ground truth to score against.
