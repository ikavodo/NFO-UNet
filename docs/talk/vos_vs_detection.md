# Video Object Segmentation vs. Detection: The Case of Through-Foliage Sensing

Nov 1, 2026 · @Ido Akov

## Summary

Detection outputs a class label and an axis-aligned box per frame. Semi-supervised video object segmentation (VOS) outputs a pixel mask of an arbitrary, user-specified object, propagated through time with a consistent identity. VOS is the right tool when the box is the wrong geometric primitive, when the target is not a predefined category, or both.

Through-foliage sensing is close to the worst case for boxes and the best case for VOS. The target is visible only as scattered fragments, no single frame suffices, and evidence must be integrated over time at the pixel level.

## Where bounding boxes break down

- **Thin, non-convex, or articulated objects** (cables, tools, limbs, animals, bicycles). The box is mostly background, so features, area and centroid computed from it are contaminated.
- **Occlusion and clutter.** Overlapping objects give overlapping boxes, and a box may contain more occluder than target. A mask assigns each pixel to one instance, which also makes cross-frame association more reliable.
- **Deformable or amorphous objects** (cloth, fluids, smoke, cells, organs), where shape change is itself the signal of interest.

## Use cases

Masks are required wherever the output depends on object geometry rather than rough location.

| Domain | Why a mask is needed |
| --- | --- |
| Video editing and VFX | Rotoscoping, object removal and inpainting, background replacement, per-object grading need exact boundaries |
| AR compositing | Rendering virtual content behind or around a real object needs its per-frame silhouette |
| Robotic manipulation | Grasp planning, contact points and affordances depend on shape, especially in clutter and with deformables |
| Medical and surgical video | Instrument tracking and skill assessment; lesion or tissue area, volume and boundary |
| Biology and science | Cell or organism tracking with morphology (division, shape change); fluid and particle phenomena |
| Measurement in general | Area, shape, deformation or precise position over time (biomechanics, industrial inspection) |
| Class-agnostic tracking | Click or scribble on any object in frame 1 and follow it, without it being in a label set |

Detection suffices for counting, coarse localization, surveillance and traffic analytics, and multi-object tracking for trajectories or re-identification. There a box is an adequate summary and is far cheaper to annotate and run.

The line has blurred: promptable models such as SAM 2 merge box prompting, segmentation and propagation into one pipeline, while per-frame instance segmentation (Mask R-CNN style) gives masks without class-agnostic targets or temporal propagation. The meaningful contrast today is an arbitrary target with temporally coherent pixel-level identity versus per-frame categorical localization.

## Through-foliage sensing

Six properties of the problem favor pixel-level, temporally integrated segmentation over per-frame detection.

1. **Fragmented visibility makes the box uninformative.** Under dense canopy a person or vehicle appears as scattered patches through gaps in the leaves. The box around them is mostly foliage, so pooled features are dominated by the occluder, box regression is ill-posed, and IoU-based training and evaluation are noisy. A modal mask of the visible pixels is the only well-defined output.
2. **No single frame suffices.** Per-frame detection thresholds weak evidence and discards it before it can be combined, the classic failure of detect-then-track at low SNR. VOS propagates pixel-level identity, so partial glimpses from different viewpoints (camera parallax) or moments (wind-driven gaps) can be associated and integrated. This is effectively track-before-detect at the pixel level, and the union over time can be far more complete than any frame.
3. **Motion is more reliable than appearance.** Appearance suffers from heavy occlusion, illumination dapple and small target size. Motion separates three layers per pixel: target motion (self-motion plus parallax at the target's depth), foliage motion (stochastic, at a different depth) and background. Grouping pixels by coherent motion is a segmentation problem, not a box problem.
4. **Masks improve synthetic aperture integration.** In airborne synthetic aperture or integral imaging (e.g. airborne optical sectioning with drones, often thermal), views are registered to a focal plane and averaged to blur out foliage. Residual occluder pixels limit contrast; per-pixel occluder and target masks allow weighting or excluding occluder samples for a cleaner integral image.
5. **Intermittent visibility needs persistent identity.** Fragments appear and disappear as gaps open and close. A detector loses the target whenever visibility drops; a memory-based VOS model keeps the object representation alive and re-associates new fragments. Amodal estimation of full extent from accumulated evidence is the natural extension.
6. **Downstream tasks need precise geometry.** Search and rescue, wildlife monitoring and surveillance need exact position, posture (lying vs. standing matters for casualty assessment) and size, all derived from the visible pixel set rather than a box padded with leaves.

## Caveats

Ground truth is the main obstacle. Fragment-level modal masks are tedious and ambiguous to annotate, and amodal ground truth is often unobtainable without a controlled setup, such as capturing a scene with and without foliage, or synthetic data.

Evaluation needs care too. Mask IoU on tiny fragmented regions is unstable, so evaluate on the temporally accumulated mask or on recall of target pixels, not per-frame IoU alone.
