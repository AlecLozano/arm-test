# Arm Control

This document explains how [Arm-Sim.py](Arm-Sim.py) represents and drives the Franka Emika
Panda arm using the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)
(`roboticstoolbox`), and how an Xbox controller's input is turned into
end-effector motion. The concepts here are drawn from the accompanying
tutorial notebooks [1 Manipulator Kinematics.ipynb](1%20Manipulator%20Kinematics.ipynb)
and [3 Resolved-Rate Motion Control.ipynb](3%20Resolved-Rate%20Motion%20Control.ipynb).

## 1. How the Robotics Toolbox represents the robot

### 1.1 Elementary Transforms (ETs) and the ETS

The Robotics Toolbox describes a manipulator's kinematics with an
**Elementary Transform Sequence (ETS)**. An ETS is simply an ordered chain of
**Elementary Transforms (ETs)** — small translations or rotations along/about
a single axis — multiplied together from the robot's base frame out to its
end-effector:

```
E1 * E2 * E3 * ... * Em
```

Each ET is one of six types: a translation along x/y/z (`tx`, `ty`, `tz`) or a
rotation about x/y/z (`Rx`, `Ry`, `Rz`). An ET's parameter can be either:

* a **constant** — a fixed offset baked into the robot's geometry (e.g. the
  physical length of a link), or
* a **joint variable** — `q(t)`, the live angle (revolute joint) or
  displacement (prismatic joint) that changes as the robot moves.

For example, the Panda's third link is built from the ET chain
`SE3(0, -0.316, 0; 90°, -0°, 0°) ⊕ Rz(q2)`: a fixed offset/rotation
representing the physical shape of the link, followed by a variable rotation
about z driven by joint `q2`.

The forward kinematics of the whole arm — the end-effector pose `T_e` in the
base frame — is just the product of every ET in the chain, each evaluated at
its current joint value:

```
T_e(q) = E1(η1) * E2(η2) * ... * Em(ηm)
```

`roboticstoolbox` computes this for you via `panda.fkine(q)` (returns a
`spatialmath.SE3` pose object) or `panda.eval(q)` (returns a plain 4x4 numpy
array), as seen in [Arm-Sim.py:118](Arm-Sim.py#L118):

```python
Te = panda.fkine(panda.q).A   # .A extracts the raw 4x4 matrix from the SE3 object
```

### 1.2 Links, joints, and the Robot tree

An ETS alone only carries kinematic data — it says nothing about mass,
inertia, or geometry. The toolbox layers a `Robot` class on top of ETS to
capture the whole arm:

* Each `Link` wraps one segment's ETS along with its dynamic/visual data
  (mass, inertia, mesh, joint limits, etc.). A link's ETS is the sequence of
  ETs connecting it to its parent link.
* A `Robot` (the Panda is built via `rtb.models.Panda()`) holds a **tree** of
  `Link` objects — each link has at most one parent but can have several
  children — so the toolbox can represent serial-link arms as well as
  branched mechanisms.
* An end-effector or gripper is represented separately as a `Gripper`, listed
  under `Robot.grippers`.

Printing the robot shows this structure directly:

```
ERobot: panda (by Franka Emika), 7 joints (RRRRRRR), 1 gripper, geometry, collision
┌──────┬──────────────┬───────┬─────────────┬─────────────────────────────────────┐
│ link │     link     │ joint │   parent    │        ETS: parent to link          │
├──────┼──────────────┼───────┼─────────────┼─────────────────────────────────────┤
│    0 │ panda_link0  │       │ BASE        │ SE3()                               │
│    1 │ panda_link1  │     0 │ panda_link0 │ SE3(0, 0, 0.333) ⊕ Rz(q0)           │
│    2 │ panda_link2  │     1 │ panda_link1 │ SE3(-90°, -0°, 0°) ⊕ Rz(q1)         │
│  ... │     ...      │  ...  │     ...     │                 ...                 │
│    8 │ @panda_link8 │       │ panda_link7 │ SE3(0, 0, 0.107)                    │
└──────┴──────────────┴───────┴─────────────┴─────────────────────────────────────┘
```

Each row is a `Link`; the "ETS: parent to link" column is that link's local
ETS — the fixed offset that places the link, `⊕`'d with the variable
rotation/translation contributed by its joint (`Rz(q0)`, `Rz(q1)`, …). The
Panda has 7 revolute joints (`RRRRRRR`), so its overall configuration is a
7-element vector `q = [q0, q1, ..., q6]`, exposed as `panda.q`.

The robot's joint limits are stored per-joint in `panda.qlim` (a `2 x n`
array of `[lower; upper]` bounds in radians), and named configurations such
as the "ready" pose (`panda.qr`) and "zero" pose (`panda.qz`) are stored as
convenience presets.

## 2. How the robot moves the end-effector

### 2.1 The manipulator Jacobian

Differentiating the forward-kinematics equation with respect to time relates
joint velocity to end-effector (Cartesian) velocity through the **manipulator
Jacobian** `J(q)`:

```
v = J(q) * qd
```

where `v` is a 6-element spatial velocity `[vx, vy, vz, wx, wy, wz]`
(translational velocity followed by angular velocity), and `qd` is the
vector of joint velocities.

`roboticstoolbox` computes `J(q)` for the current configuration with:

* `panda.jacob0(q)` — the Jacobian expressed in the **base frame**.
* `panda.jacobe(q)` — the Jacobian expressed in the **end-effector frame**.

Both return a `6 x n` matrix (`6 x 7` for the Panda's 7 joints).

### 2.2 Resolved-Rate Motion Control (RRMC)

To move the end-effector at a *desired* velocity, we need the inverse
relationship — given a desired Cartesian velocity `ev`, solve for the joint
velocities `qd` that produce it:

```
qd = J(q)^-1 * ev
```

This only works directly when `J` is square (a 6-DOF arm). The Panda has 7
joints, so `J` is `6 x 7` and not invertible. Instead, [Arm-Sim.py](Arm-Sim.py)
uses the **Moore-Penrose pseudoinverse**, which yields the joint-velocity
solution with the smallest overall joint speed among all solutions that
achieve `ev`:

```python
J = panda.jacob0(panda.q)      # base-frame Jacobian at the current pose
J_pinv = np.linalg.pinv(J)     # pseudoinverse, since J is 6x7 (not square)
panda.qd = J_pinv @ ev         # joint velocities that realize the desired ee velocity
```

Setting `panda.qd` and calling `env.step(dt)` integrates those joint
velocities forward by `dt` seconds each loop iteration, producing continuous
end-effector motion in the direction of `ev`. Because the Jacobian is
recomputed from `panda.q` every iteration, this steers the end-effector
along a (locally) straight-line path even as the arm's configuration
changes.

[Arm-Sim.py](Arm-Sim.py) applies two safety clamps on top of this basic loop:

* **Floor limit** — if the end-effector's z position (`Te[2, 3]`) drops below
  10 cm, any further downward velocity command (`ev[2] < 0`) is clamped to
  zero, preventing the arm from driving itself into the ground.
* **Joint limits** — after each step, `panda.q` is clipped to `panda.qlim`
  so no joint is driven past its mechanical range of motion.

## 3. How the controller affects the speed

### 3.1 Reading controller input

[Arm-Sim.py](Arm-Sim.py) reads an Xbox controller through the
`ControllerCommon.XboxController` helper. Each call to
`controller.getControllerInput()` returns any new axis/button/D-pad events as
`"id:value"` strings, which are split into `id` and `val` with
`getInputID()` / `getInputValue()`. Analog axes (triggers and sticks) report
`val` in the range `-1.0` to `1.0`.

Specific input IDs are mapped to specific motions:

| Input ID | Meaning              | Effect                                   |
|----------|-----------------------|-------------------------------------------|
| `6`      | Left stick axis        | Sets desired x-axis velocity `ev[0]`     |
| `5`      | Left stick axis        | Sets desired y-axis velocity `ev[1]`     |
| `8`      | Trigger/axis            | Sets desired z-axis velocity `ev[2]` (inverted) |
| `9`      | Trigger                | Drives the wrist joint forward           |
| `10`     | Trigger                | Drives the wrist joint backward          |

### 3.2 Speed caps turn stick position into velocity

The raw axis value (`-1` to `1`) is not used directly as a velocity — it is
scaled by fixed **speed cap** constants so that full stick/trigger deflection
corresponds to a bounded, safe top speed:

```python
speedCap = .3          # max linear ee speed percentage, along x/y/z
wrist_speed_cap = .7   # max wrist joint speed percentage
```

* **Translational velocity** — for x, y, and z, the axis value is multiplied
  directly by `speedCap`:

  ```python
  ev[0] = val * speedCap        # x-axis velocity
  ev[1] = val * speedCap        # y-axis velocity
  ev[2] = val * -1 * speedCap   # z-axis velocity (inverted so pushing the axis moves up)
  ```

  So `ev`'s translational components scale linearly from `0` at a centered
  stick to `±speedCap` (30%) at full deflection. This `ev` vector is the
  same desired end-effector velocity fed into the RRMC pseudoinverse
  calculation described in Section 2.2 — the speed cap therefore directly
  bounds how fast the end-effector can be commanded to move in Cartesian
  space.

* **Wrist velocity** — the wrist is driven independently of the Jacobian
  solution. The trigger's `-1..1` value is first remapped to `0..1` (via
  `(1 + val) / 2`) and then scaled by `wrist_speed_cap`:

  ```python
  wrist_speed = (1 + val) / 2 * wrist_speed_cap          # forward (id 9)
  wrist_speed = (1 + val) / 2 * wrist_speed_cap * -1      # backward (id 10)
  ```

  This produces a wrist speed between `0` and `±0.7` rad/s. Rather than
  going through the Jacobian, this value directly overwrites joint 6's
  (index `5`) computed velocity after the pseudoinverse solve:

  ```python
  if wrist_speed != 0:
      panda.qd[5] = wrist_speed   # fixed velocity, not accumulating
  ```

  This means the wrist rotates at a constant commanded rate independent of
  whatever joint-6 velocity RRMC would otherwise have produced to satisfy the
  Cartesian translation — it overrides that entry of `qd` outright.

### 3.3 Putting it together each loop iteration

Each iteration of the main loop in [Arm-Sim.py](Arm-Sim.py):

1. Reads new controller events and updates `ev` (Cartesian velocity) and
   `wrist_speed` according to the mappings and caps above.
2. Computes the current Jacobian and its pseudoinverse from `panda.q`.
3. Applies the floor-height safety clamp to `ev[2]`.
4. Solves `panda.qd = J_pinv @ ev` for the joint velocities that realize the
   capped Cartesian velocity.
5. Clips `panda.q` to the joint limits (`panda.qlim`).
6. Overwrites the wrist joint's velocity with `wrist_speed` if nonzero.
7. Steps the Swift simulator forward by `dt = 0.05` seconds, integrating the
   commanded joint velocities into a new pose.

The net effect is that `speedCap` and `wrist_speed_cap` are the two knobs
that determine overall arm speed: `speedCap` bounds how fast the
end-effector translates through space (via the RRMC/Jacobian pipeline), and
`wrist_speed_cap` bounds how fast the wrist spins (applied directly to joint
velocity, bypassing the Jacobian).
