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

Helical (screw) symmetry has been fully implemented in `models/rfd3/src/rfd3/inference/symmetry/frames.py` and validated for fiber generation.

### Logic
A new function `get_helical_frames(handedness, radius, monomers_per_turn, rise_per_turn, num_turns)` generates rotation matrices ($R$) and translation vectors ($T$) for a helical assembly.

**Key features:**
*   **Handedness:** Supports 'R' (Right) and 'L' (Left). Right-handed is defined as clockwise rotation (negative `d_phi`).
*   **Radius:** Encoded in the ASU coordinates (frames apply rotation and axial translation only).
*   **Geometry:**
    Subunit $i$ is positioned at:
    *   Angle: $\theta_i = i \times \frac{2\pi}{\text{MonomersPerTurn}}$
    *   Height: $Z_i = i \times \frac{\text{RisePerTurn}}{\text{MonomersPerTurn}}$
    *   Translation: $T_i = [0, 0, Z_i]$
    *   Transform 0 is identity ($R = I$, $T = 0$).

**Warning (Radius currently does effectively nothing in frames):**
The H `radius` parameter is parsed but not used in frame translations. It only affects initialization if the ASU coordinates are explicitly offset before diffusion. If you do not seed the ASU with a radius, the frames alone will not create one.

### Implementation Details
*   **File:** `models/rfd3/src/rfd3/inference/symmetry/frames.py`
    *   Added `get_helical_frames`.
    *   Updated `get_symmetry_frames_from_symmetry_id` to parse the new `H_` prefix.

### Differences vs C/D That Can Explain Current H Issues
These differences are structural (code-level) and can lead to the behavior you described:

1. **Transform 0 is identity for H.**
    - For C/D, transform 0 is identity. For H, transform 0 is also identity (no translation); axial translation is applied in frames for other subunits only.
    - This avoids repeated translation of the ASU during symmetrization.

2. **Per-step COM centering interacts with translation.**
    - `apply_symmetry_to_xyz_atomwise` subtracts the mean of all non-fixed atoms before applying transforms (except in partial diffusion).
    - For H, COM centering is now skipped when symmetry transforms include translation, to avoid re-centering and re-applying axial offsets on every step.

3. **Symmetry is applied only to denoised coordinates, not noisy ones.**
    - The model denoises `X_noisy_L` without symmetry, then the output is symmetrized.
    - This can lead to rotational drift unless the model learns to counteract it.

4. **Symmetry ID truncation risk.**
    - Symmetry IDs are stored in an atom-array annotation with dtype `U6`.
    - Long helical IDs such as `H_R_20.0_4.5_20.0_3` will be truncated in annotations, which can break code paths that rely on `atom_array.symmetry_id[0]` for reconstruction or validation.

5. **Debug prints in `get_helical_frames`.**
    - The function prints per-call debug output. This is harmless but adds noise and makes it harder to validate behavior in batch runs.

### Usage
To run helical symmetry, use the following ID format in your YAML/JSON input:

`H_{Handedness}_{Radius}_{MonomersPerTurn}_{RisePerTurn}_{NumTurns}`

**Arguments:**
1.  **Handedness**: `R` or `L`. (e.g., `R`)
2.  **Radius**: Radius of the helix in Angstroms (e.g., `20.0`). *Currently used only for ASU seeding; frames ignore it.*
3.  **MonomersPerTurn**: Number of subunits per 360-degree turn (e.g., `4.5`).
4.  **RisePerTurn**: Vertical rise per full 360-degree turn in Angstroms (e.g., `20.0`).
5.  **NumTurns**: Total number of turns to generate (determines the fiber length).

**Example YAML:**
```yaml
uncond_Helical_example:
  length: 100
  symmetry:
    id: "H_R_20.0_4.5_20.0_3"
```
This generates a Right-handed helix with a 20Å radius, 4.5 monomers per turn, 20Å rise per turn, spanning 3 turns.

## 8. Trajectory Output Notes (Important)

- The inference engine currently swaps denoised and noisy trajectories when assembling stacks for output. Files labeled `_noisy_model_*.cif` can contain denoised (symmetrized) coordinates, and vice versa.
- If you are validating symmetry using noisy trajectories, confirm the swap in [models/rfd3/src/rfd3/engine.py](models/rfd3/src/rfd3/engine.py) before interpreting results.
