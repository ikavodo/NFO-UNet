# Second annotator: CVAT setup and annotation guide (2026-10-21)

You will label, frame by frame, the pixels where a person is **directly visible** through dense foliage. Your labels
are compared with a first annotator's (inter-annotator agreement), so please work **independently**: do not look at
anyone else's labels or at any model output for these frames.

You receive from the project lead: `annot2_seq2_tierA.zip`, `annot2_seq3_tierA.zip`, `annot2_seq4_tierA.zip`
(tier A, 23 frames, do these first), later the three `*_tierB.zip` (23 more frames), and `labels.json`.

## 1. Install CVAT

Two options. **Use A (local)** unless the project lead says otherwise: the frames never leave your machine.
Use **Google Chrome**, the only browser CVAT supports.

### A. Local CVAT with Docker (recommended)

Prerequisites: Docker with Docker Compose, Git, Chrome.
- Linux: Docker Engine + the compose plugin (`docker compose version` must work).
- Windows: Windows 10 2004 or newer, WSL2, Docker Desktop with the WSL2 backend; run the commands in the WSL2 shell.
- macOS: Docker Desktop for Mac, Git (Xcode Command Line Tools).

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d                       # first start downloads the images: several GB, a few minutes
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'   # choose a user name + password
```

Open **http://localhost:8080** in Chrome and log in with that user.

- Stop CVAT: `docker compose down` (in the `cvat` folder). Your tasks and labels are kept in Docker volumes and are
  there again after the next `docker compose up -d`. **Never** run `docker compose down -v`: `-v` deletes them.
- If `docker compose up` complains that port 8080 is in use, stop the other program using it.

### B. CVAT online (app.cvat.ai)

Create a free account at https://app.cvat.ai and log in. Everything below is the same. Only use this if the project
lead confirms the frames may be uploaded to a third-party server.

## 2. Create the tasks (one per zip, tier A first)

For each zip:

1. **Tasks → + → Create a new task.**
2. **Name:** the zip name without `.zip`, e.g. `annot2_seq2_tierA`.
3. **Labels:** switch to the **Raw** tab, delete its content, paste the whole content of `labels.json`, click
   **Done**. You should see two labels: `person` (pink) and `void` (yellow), both of type *mask*.
4. **Select files → My computer:** choose the zip.
5. **Advanced configuration:** Image quality **100**. Leave sorting at the default (the file names are the frame
   numbers, e.g. `00052.jpg`).
6. **Submit & Open.** Do **not** upload or import any annotations: the task starts empty on purpose.

Expected frame counts: seq2 tier A 8, seq3 tier A 7, seq4 tier A 8.

## 3. Annotate

Open the task → open its **Job** (link in the Jobs list). Per frame:

1. **What to paint (`person`):** only the pixels where the person (body, hair, clothing, carried object) is
   directly visible through gaps in the leaves.
   - Do not fill in hidden body parts: if a leaf covers the arm, the leaf is not person.
   - Shadows are background.
2. **`void`:** pixels you genuinely cannot decide (a leaf edge over the body, motion blur). Use it sparingly; void
   pixels are excluded from scoring. Paint void *on top of* person where they overlap.
3. **Tools:** in the left toolbar choose the **mask / brush** tool ("Draw new mask"), select the label, paint;
   use the eraser to correct; press **Done** (or N) to finish the mask. Brush size: adjust in the top bar.
   **No AI or assisted tools** (magic wand, interactors, SAM, "AI tools" tab): only brush and eraser.
4. Zoom (mouse wheel) as much as you need. Use the frame slider to look at neighbouring frames if it helps you
   decide, but label only the frames in the task.
5. **Save** often (Ctrl+S). When a frame is finished, move on and do not go back to change it later.

There is no time limit. If you are unsure about the rules, ask the project lead before starting, not halfway through.

## 4. Export and send back

When a task is finished: **Tasks → ⋮ (Actions) of the task → Export task dataset**:
- Format: **CVAT for images 1.1**
- **Save images: off** (annotations only)

You get a zip containing `annotations.xml`. Send the three tier A zips (named after their task, e.g.
`annot2_seq2_tierA_export.zip`) to the project lead. Then the same for tier B.

## For the project lead (import, nothing is overwritten)

Unpack each export into `data/cvat/exports/annot2_seq<s>_tier<T>/` and run, from the repo root:

```bash
PYTHONPATH=. python -m benchmark.nfo_vos.tools.cvat_io import data/cvat/exports/annot2_seq2_tierA/annotations.xml handAv2 seq2
```

(the same for seq3 and seq4) → `data/nfo_processed/seq<s>_gt/<frame>_handAv2.png` (1 person, 2 void). Commit them
with `git add -f` (data/ is ignored). The task zips and `labels.json` are in `data/cvat/annot2/` (not in git).
