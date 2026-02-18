# RFdiffusion3 Symmetry Implementation Analysis

This document details the implementation of symmetry in RFdiffusion3 (rfd3) within the Foundry codebase. It covers the file structure, handling logic, integration with the diffusion process, and a guide for extending the system with new symmetry types such as screw symmetry.

## 1. Key Codebase Files

The implementation of symmetry is modularized across inference utilities, frame generation, and the diffusion sampler.

### Core Logic
- **`models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py`**
  - **Function:** `apply_symmetry_to_xyz_atomwise`: The workhorse function that applies rigid body transformations (Rotation $R$ and Translation $T$) to the Asymmetric Unit (ASU) to generate the full complex.
  - **Function:** `make_symmetric_atom_array`: Used during initialization to setup the initial symmetric state from input designs.
  - **Class:** `SymmetryConfig`: Data structure dealing with configuration parameters.

- **`models/rfd3/src/rfd3/inference/symmetry/frames.py`**
  - **Role:** The geometry engine. It calculates the specific $3 \times 3$ Rotation matrices and $1 \times 3$ Translation vectors for a given symmetry group ID.
  - **Key Functions:** `get_cyclic_frames`, `get_dihedral_frames`.
  - **Entry Point:** `get_symmetry_frames_from_symmetry_id`.

### Diffusion & Sampling
- **`models/rfd3/src/rfd3/model/inference_sampler.py`**
  - **Class:** `SampleDiffusionWithSymmetry`
    - **Role:** Overrides the standard sampling loop to enforce symmetry constraints on the *denoised* coordinates during most of the reverse diffusion steps.

- **`models/rfd3/src/rfd3/transforms/symmetry.py`**
    - **Role:** Converts atom-array symmetry annotations (`sym_transform_Ori`, `sym_transform_X`, `sym_transform_Y`) into runtime transforms (`sym_transform`) used by the sampler.

### Input Parsing
- **`models/rfd3/src/rfd3/inference/input_parsing.py`**
  - **Role:** Parses user input (e.g., "C3", "D2") and initializes the `SymmetryConfig`.

## 2. Symmetry Logic and Handling

Symmetry in RFdiffusion3 is handled via an **explicit symmetrization** strategy. The model does not inherently "know" symmetry in its weights; instead, the inference process forces the generated coordinates to be symmetric at every step.

### The Mechanism
1.  **Frame Generation (`frames.py`):**
    When a job starts, the symmetry ID (e.g., "C3") is resolved into a list of rigid body frames.
    -   A "frame" consists of a tuple: `(RotationMatrix, TranslationVector)`.
    -   For point groups like Cyclic (Cn) or Dihedral (Dn), the TranslationVector is usually zero `[0,0,0]`.
    -   These frames act as operators: $X_{full} = \{ R_i \cdot X_{ASU} + T_i \mid i \in \text{subunits} \}$.

2.  **Feature Storage (`transforms/symmetry.py`):**
    The per-atom annotations `sym_transform_Ori`, `sym_transform_X`, `sym_transform_Y` are converted to $R, T$ frames via `framecoords_to_RTs` and stored in the features dict as `sym_transform`.

3.  **Atom-wise Application (`symmetry_utils.py`):**
    The function `apply_symmetry_to_xyz_atomwise` performs the actual tensor operations using `torch.einsum` for rotation and simple addition for translation.
    - A *center-of-mass correction* is applied on every call for non-partial diffusion: the mean of all non-fixed atoms is subtracted before symmetry is applied.
    - For partial diffusion, this centering is skipped.

## 3. Integration with the Diffusion Process

The symmetry logic acts as a constraint applied `post-hoc` to the network predictions at each step of the reverse diffusion process.

### The Algorithm (`inference_sampler.py`)
In `SampleDiffusionWithSymmetry`, the sampling loop proceeds as follows:

1.  **Denoise Step:** The neural network predicts the noise/update for the current coordinates.
2.  **Symmetrization (`apply_symmetry_to_X_L`):**
    Before the next step is finalized, the **denoised** coordinates are explicitly symmetrized for most of the steps.
    -   The system takes the coordinates of the **first subunit** (the Asymmetric Unit, ASU).
    -   It discards the predicted coordinates of the other subunits.
    -   It regenerates the other subunits by applying the symmetry frames ($R, T$) to the ASU.
    
    $$ X_{subunit_i} = R_i \cdot X_{ASU} + T_i $$

This ensures that no matter what small numerical deviations the network predicts for essentially identical subunits, they are forced back into perfect symmetry before the next iteration.

### How Often Symmetry Is Applied
Symmetry is applied on the *denoised* structure only and only while $c_t > \gamma_{min\_sym}$:

- The sampler computes $\gamma_{min\_sym}$ as the noise schedule value at index `int(len(noise_schedule) * sym_step_frac)`.
- With the default `sym_step_frac=0.9`, this means symmetry is applied for almost all steps, except the final tail of the schedule.
- The **noisy** structure (`X_noisy_L`) is never symmetrized by the sampler. Note: the current inference output assembly swaps noisy and denoised trajectories when saving; see the trajectory notes below.

## 4. Inference Execution

1.  **Entry:** `run_inference.py` calls `inference/input_parsing.py`.
2.  **Configuration:** The parsing logic detects the symmetry argument.
3.  **Initialization:** `make_symmetric_atom_array` builds the initial placeholders.
4.  **Sampling:** The `RFD3InferenceEngine` initializes the `SampleDiffusionWithSymmetry` class.
5.  **Runtime:** The `sym_transforms` are passed into the model forward pass (via feature dict `f`), allowing the sampler to access $R$ and $T$ matrices without recomputing them.

## 5. Cyclic (C) and Dihedral (D) Symmetry: Radius, Clashes, and Expansion

### Is Radius Pre-defined?
No. For C and D symmetries the translation component is always zero (`T = [0, 0, 0]`), so the symmetry operation is a pure rotation about the Z-axis.

- The *effective radius* of the assembly is simply the distance of the ASU atoms from the origin.
- The origin is defined by `set_com(...)` in input parsing (unless partial diffusion with symmetry is used, in which case COM centering is skipped).
- The diffusion model determines how far the ASU drifts from the origin during denoising. Symmetry does not introduce a radius parameter for C/D.

### How Are Clashes Avoided / Expansion Handled?
There is **no explicit clash avoidance** or radial expansion logic in the symmetry layer.

- Symmetry is enforced by *overwriting* all non-ASU subunits with a transformed ASU.
- Any clash avoidance or size expansion comes from the diffusion model itself and its learned priors, not from the symmetry system.
- The symmetry layer only ensures *geometric consistency* between subunits.

### Quality-of-life Features Present for C/D (Also Used by H)
These are general symmetry utilities that help C/D workflows and are reused by H, but they do not add special helical behavior:

- **Motif exclusion:** Unsymmetrized motifs and ligands are removed from the symmetry expansion and re-attached later.
- **Fixed motif handling:** Non-indexed motifs are given a fixed transform id (`-1`) and are excluded from COM centering.
- **2D conditioning support:** 2D conditioning annotations are re-indexed per symmetry unit.

## 6. Adding New Symmetries (e.g., Screw Symmetry)

To add Screw symmetry (Helical symmetry), you do not need to modify the diffusion core or the `symmetry_utils.py` logic, as the system already supports arbitrary Rotation+Translation.

### The Best Place to Edit: `models/rfd3/src/rfd3/inference/symmetry/frames.py`

This is the only file requiring significant logic changes.

### Steps to Implement:

1.  **Create a Generator Function:**
    Add a function like `get_screw_frames(n_subunits, rotation_angle, translation_rise, ...)` in `frames.py`.
    -   **Rotation:** Calculate $R$ similarly to `get_cyclic_frames` (rotation around Z-axis).
    -   **Translation:** THIS is the key difference.
        -   For subunit $i$:
        
        $$ T_i = [0, 0, i \times \text{rise}] $$
        
    -   Return list of tuples: `[(R_0, T_0), (R_1, T_1), ...]`.

2.  **Register the ID:**
    Update `get_symmetry_frames_from_symmetry_id` to parse a new ID format (e.g., "H" or "S").
    ```python
    # Hypothetical implementation structure
    if sym_id.startswith("H"):
        # parsing logic for "H_rot_rise"
        return get_screw_frames(...)
    ```

3.  **Verify Configuration:**
    Ensure that `SymmetryConfig` can pass any necessary floating-point parameters (like specific rise values) if they aren't encoded in the string ID strings.

### Summary
The RFdiffusion3 system is highly flexible. It decouples the geometric definition of symmetry (Matrices in `frames.py`) from the application of symmetry (Tensor ops in `symmetry_utils.py`). Adding space group or helical symmetries is reduced to a geometric problem of defining the correct list of standard transformations.

## 7. Implemented Helical Symmetry

Helical (screw) symmetry has been fully implemented and validated for fiber generation.

### Symmetry ID Format

`H_{Handedness}_{Radius}_{AnglePerMonomer}_{RisePerMonomer}_{NumMonomers}`

| Parameter | Description |
|---|---|
| **Handedness** | `R` (right-handed, clockwise) or `L` (left-handed, counter-clockwise) |
| **Radius** | Distance of the ASU center-of-mass from the helical axis (Z) in Ångströms. Not used in frames — enforced via coordinate initialization and per-step radius clamping. |
| **AnglePerMonomer** | Rotation per monomer in **degrees** (e.g., `60` = 6 monomers per full turn). |
| **RisePerMonomer** | Axial (Z) translation per monomer in Ångströms. |
| **NumMonomers** | Total number of monomers (subunits) in the assembly. Truncated to `int`. |

### ASU Placement (Coordinate Space)

The **Asymmetric Unit (ASU)** is the first subunit (transform 0, chain A). It is the master copy; all other subunits are generated from it.

**Coordinate system:** The helical axis is always **Z**. The ASU is placed in the **XZ-plane**:

1. During input parsing (`_set_origin` in `input_parsing.py`), for helical symmetry:
   - COM centering is **skipped** — input coordinates are preserved as-is for conditioned runs.
   - For non-fixed (diffused) atoms, coordinates are zeroed out: `coord = 0.0`.
   - `apply_helical_asu_radius_offset` then sets the X-coordinate of all non-fixed ASU atoms to the radius value:
     $$ \text{coord}_{ASU,\text{non-fixed}} = (\text{radius}, 0, 0) $$
   - This places the ASU's center-of-mass at distance `radius` from the Z-axis, along the +X direction.

2. During diffusion (`apply_symmetry_to_xyz_atomwise` in `symmetry_utils.py`):
   - At every symmetry step, the ASU's XY center-of-mass is **re-clamped** to the target radius. The code computes the current COM in XY, scales it so `||COM_xy|| == radius`, and shifts all ASU atoms by the delta. This prevents the noise schedule from overwhelming the radius seed.
   - **No COM centering** is applied for helical symmetry (neither pre- nor post-symmetrization), unlike C/D where the full assembly is re-centered at the origin each step.

### Frame Generation and Handedness

`get_helical_frames` in `frames.py` generates one $(R_i, T_i)$ pair per monomer.

**Rotation** is about the Z-axis. The angular step is:

$$ d\phi = \text{AnglePerMonomer} \text{ (in radians)} $$

- **Left-handed (`L`):** `d_phi` stays **positive** → counter-clockwise rotation when viewed from +Z.
- **Right-handed (`R`):** `d_phi` is **negated** → clockwise rotation when viewed from +Z.

**Translation** is purely axial (+Z):

$$ T_i = [0, 0, i \times \text{RisePerMonomer}] $$

**Frame 0 is always identity** ($R_0 = I_{3 \times 3}$, $T_0 = [0,0,0]$). This means the ASU itself is never transformed — only copies $i \geq 1$ receive non-trivial transforms.

**Example for `H_L_20.0_60_10.0_3` (3 monomers):**

| Monomer $i$ | Rotation $\theta_i$ | Translation $T_i$ |
|---|---|---|
| 0 (ASU) | $0°$ | $[0, 0, 0]$ |
| 1 | $+60°$ (CCW) | $[0, 0, 10]$ |
| 2 | $+120°$ (CCW) | $[0, 0, 20]$ |

**Example for `H_R_20.0_60_10.0_3` (same but right-handed):**

| Monomer $i$ | Rotation $\theta_i$ | Translation $T_i$ |
|---|---|---|
| 0 (ASU) | $0°$ | $[0, 0, 0]$ |
| 1 | $-60°$ (CW) | $[0, 0, 10]$ |
| 2 | $-120°$ (CW) | $[0, 0, 20]$ |

### Which Chain Is Copied

- The ASU is **chain A** (transform_id = 0). It carries the annotation `is_sym_asu = True`.
- During initialization (`make_symmetric_atom_array`), the ASU is duplicated once per frame. Each copy gets a new chain ID (B, C, D, …) via `reannotate_chain_ids`, and its coordinates are transformed by `apply_symmetry_to_atomarray_coord(copy, frame)` using `coord = coord @ R + T`.
- During diffusion, `apply_symmetry_to_xyz_atomwise` reads only the ASU coordinates (`X_L[:, is_sym_asu, :]`) and overwrites all other subunits:

$$ X_{\text{subunit}_i} = X_{ASU} \cdot R_i + T_i $$

(Note: the codebase stores $R^T$ so that the row-vector multiplication `coord @ R_stored` yields the correct result.)

- **Unindexed motifs and ligands** (e.g., HEM) are excluded from symmetry expansion — they are stripped out before duplication, given `transform_id = -1` and `entity_id = -1`, and appended back at the end with lowercase chain IDs.

### Unconditional Helical Design

**Config:**
```yaml
L_unconditional_NEW_nomenclature:
  length: 80
  is_non_loopy: true
  symmetry:
    id: "H_L_20.0_60_10.0_3"
```

**What happens step-by-step:**

1. **No input PDB.** An 80-residue ASU is created from scratch with no motif atoms.
2. **Origin setting:** All atom coordinates are zeroed, then `apply_helical_asu_radius_offset` sets them to $(20, 0, 0)$.
3. **Symmetry expansion:** `make_symmetric_atom_array` takes the single-chain ASU and produces 3 chains:
   - Chain A (ASU, identity transform)
   - Chain B (rotated +60°, translated +10Å along Z)
   - Chain C (rotated +120°, translated +20Å along Z)
   - Since there are no motifs (`is_symmetric_motif` has nothing to align), frames come directly from the symmetry ID via `get_helical_frames`.
4. **Diffusion loop:** At each denoising step (while $c_t > \gamma_{min\_sym}$):
   - The network denoises `X_noisy_L` (which is NOT symmetrized).
   - The denoised output `X_denoised_L` is symmetrized: the ASU's XY COM is scaled to radius 20Å, then chains B and C are overwritten by applying their respective frames to the ASU.
   - No COM centering is performed (helical mode skips it).
5. **Result:** A 3-monomer left-handed helical fiber with 80 residues per monomer, 60° rotation per step, 10Å axial rise per step, and ~20Å radial distance from the Z-axis.

### Conditioned Helical Design (Heme + Histidine)

**Config:**
```yaml
cable_L_narrow_burried:
  length: 80
  is_non_loopy: true
  input: "/home/tadas/code/rfd3_fibers/inputs/heme_aligned_his_atom_both_sides_H_L_20_20_25_4.pdb"
  ligand: HEM
  unindex: "A87,A88"
  select_fixed_atoms:
    A87: "NE2"
    A88: "NE2"
  symmetry:
    id: "H_L_20_20_25_4"
```

**What happens step-by-step:**

1. **Input PDB loaded.** The PDB contains a pre-arranged helical assembly: multiple protein chains with heme (HEM) ligands and histidine coordination residues (A87, A88) already positioned in the correct helical geometry.
2. **COM centering skipped.** `center_symmetric_src_atom_array` detects helical symmetry and preserves all input coordinates exactly.
3. **Motif setup:**
   - `ligand: HEM` → heme atoms are marked as small molecules. They will be excluded from symmetry duplication, assigned `transform_id = -1`, and re-attached at the end.
   - `unindex: "A87,A88"` → residues 87 and 88 become **unindexed motifs** (their sequence is fixed but they are not contiguously connected to the diffused backbone).
   - `select_fixed_atoms: A87: "NE2", A88: "NE2"` → the NE2 atoms of these histidines are **spatially fixed** throughout diffusion. Their coordinates never change.
4. **Frame derivation:** Because `is_symmetric_motif` defaults to `True`, the system uses **Kabsch alignment** (`get_symmetry_frames_from_atom_array`) on the input PDB chains rather than the ID string. This derives the actual $(R, T)$ pairs by aligning each chain to the first chain, ensuring the frames match the real geometry of the input structure.
5. **Symmetry expansion:** The ASU (first chain only, after stripping HEM and unindexed motifs) is duplicated 4 times using the Kabsch-derived frames. HEM and unindexed motifs are appended back with fixed annotations.
6. **Origin setting:**
   - COM centering is skipped (helical mode).
   - Non-fixed diffused atoms are set to $(0, 0, 0)$, then offset to $(20, 0, 0)$ via `apply_helical_asu_radius_offset`.
   - Fixed atoms (NE2 of His87, NE2 of His88) **retain their input PDB coordinates**.
7. **Diffusion loop:** Same as unconditional, but:
   - Fixed atoms (NE2) are never noised and never overwritten — they act as spatial anchors.
   - The model generates the 80-residue backbone around these fixed His-NE2 atoms, respecting the helical geometry.
   - At each symmetry step, the ASU (including its fixed atoms) is copied to all 4 subunits via their frames.
   - Heme ligands remain at their input positions, unsymmetrized.
8. **Result:** A 4-monomer left-handed helical fiber with 80 residues per monomer, 20° rotation per step, 25Å rise per step, each monomer wrapping around two histidine-coordinated heme groups.

### Key Differences vs C/D Symmetry

| Aspect | C/D | H (Helical) |
|---|---|---|
| Translation component | Always $[0,0,0]$ | $[0, 0, i \times \text{rise}]$ along Z |
| COM centering per step | Yes (all axes) | **Skipped entirely** |
| Radius enforcement | N/A (distance from origin is model-determined) | ASU XY-COM is clamped to target radius every step |
| COM centering of input | Yes (centered to origin) | **Skipped** (preserves input coordinates) |
| Frame source (conditioned) | Kabsch from input PDB | Kabsch from input PDB (same) |
| Frame source (unconditional) | From symmetry ID | From symmetry ID (same) |

### Implementation Files
- **`models/rfd3/src/rfd3/inference/symmetry/frames.py`** — `get_helical_frames`: generates $(R, T)$ per monomer. `get_symmetry_frames_from_symmetry_id`: parses `H_` prefix.
- **`models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py`** — `apply_helical_asu_radius_offset`: seeds ASU at radius. `apply_symmetry_to_xyz_atomwise`: per-step radius clamping + symmetry application. `center_symmetric_src_atom_array`: skips COM centering for H.
- **`models/rfd3/src/rfd3/inference/input_parsing.py`** — `_set_origin`: helical branch that skips COM centering.
- **`models/rfd3/src/rfd3/inference/symmetry/atom_array.py`** — `get_symmetry_unit`: duplicates chain A and applies frame transforms.

### Known Caveats

1. **Debug prints.** `get_helical_frames` prints per-call debug output. Harmless but noisy in batch runs.
2. **Symmetry ID stored as `<U100`** in atom-array annotations (changed from the old `U6` to avoid truncation of long helical IDs).
3. **Symmetry applied only to denoised coords.** The noisy structure `X_noisy_L` is never symmetrized by the sampler, which can cause rotational drift.

## 8. Trajectory Output Notes (Important)

- The inference engine currently swaps denoised and noisy trajectories when assembling stacks for output. Files labeled `_noisy_model_*.cif` can contain denoised (symmetrized) coordinates, and vice versa.
- If you are validating symmetry using noisy trajectories, confirm the swap in [models/rfd3/src/rfd3/engine.py](models/rfd3/src/rfd3/engine.py) before interpreting results.
