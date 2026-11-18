# Neural manifolds in the full price space

This workflow uses neural embeddings for **both** the Heston and market sets.
All training targets contain the same 21 option-price coordinates. Neither
price set is reduced to the two directions shown in the animation.

## Construction

Each network embeds a five-dimensional latent plane into 21 dimensions, then
applies smooth invertible affine-coupling layers and a nonsingular affine map.
Positive per-coordinate output scaling preserves the embedding. There is no
price clipping. For finite weights and a nonsingular affine map, the learned
map is a smooth embedding on the open parameter domain; the closed box has
corners. This is a structural property of the learned map, not a proof that
the exact Heston pricing map is identifiable everywhere or that the market
truly follows a five-dimensional manifold.

The architecture is restored from commit `0300278`. It is implemented in
`neural_embedding.py`; training, projections and checkpoints are in
`neural_training.py`.

## Heston

C++ Monte Carlo generates 10,240 training vectors at 16,384 paths each:
8,192 Sobol parameter samples across the parameter box and 2,048 local samples
around the previously fitted parameters from the first half. These local
parameters guide sampling only; their old market targets are not training labels.
All 21 labels are freshly computed by C++.

Separate validation and test clouds each contain 512 parameter vectors with
65,536 paths and independent random streams. Training uses all 21 standardized
prices, with a strictly positive scale for every coordinate. Validation selects
the network checkpoint; the test cloud is not used to train or select it.

One Heston embedding is shared across the fixed normalized contract grid.
Every weekly fit is independently repriced with four C++ batches of 65,536
paths. The report distinguishes the learned-manifold gap, the independently
repriced C++ gap, and the neural approximation discrepancy. Additional checks
compare network derivatives with C++ finite differences and test inverse
round trips, Jacobian rank and numerical conditioning.

## Market

Each weekly market network uses only the preceding 120 complete observations.
PCA initializes the latent points and affine map; the final model is a trained
nonlinear neural embedding, with no PCA-plane fallback or price-coordinate
compression. The last 20 prior observations select training duration through
held-out reconstruction with frozen network weights. The selected duration is
then used to train on all 120 prior observations.

The current week's market grid is projected onto this learned market manifold.
That point is projected locally onto the learned Heston manifold in the full
21-price RMS metric. This is a local point-to-manifold comparison, not a
certified minimum distance between the entirety of two manifolds.

## First half and continuation

```bash
PYTHONPATH=src python3 -m hestonlab.neural_weekly --stage first-half
# Run only after authorization to continue:
PYTHONPATH=src python3 -m hestonlab.neural_weekly --stage remaining
```

The first stage ends at **105 of 209 weekly snapshots**, from 2022-01-07 through
2024-01-05. The remaining stage begins with 2024-01-12. A run lock prevents
duplicate weekly runs. C++ clouds, training state, networks, weekly numerical
results and display sections are checkpointed under `outputs/neural_weekly`.
Checksums and run identities prevent mixing different clouds or networks.
Continuation retains the Heston checkpoint and completed first-half weeks.

The current presentation remains `outputs/animation.mp4`, with matching HTML
and manifest. The previous presentation is retained in the existing
`outputs/animation-history.zip` archive. `--no-video` computes/checkpoints the
stage without exporting; `--stage render` renders its saved results.

## Display

Both sheets are evaluated directly from their neural embeddings. Local 2D
latent sections are selected only after fitting in the full 21-price space.
Newton solves align their displayed horizontal coordinates. Sections remain
inside the trained latent domain; a smaller common physical window is used
when needed to prevent a crossing caused by projection. This crop does not
establish separation of the full manifolds.

The existing presentation is retained: blue market reference, purple Heston
above it, a rotating camera at fixed distance, adaptive vertical range, 17 error
vectors, error labels revealed at one quarter, and two green tangent patches
at halfway. Two-week smoothing is applied only to display geometry and labels;
the unsmoothed neural sections and numerical results are saved separately.
