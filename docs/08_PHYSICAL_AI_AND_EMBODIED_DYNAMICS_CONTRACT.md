# Document 08: Physical AI & The Embodied Dynamics Contract

## 1. Executive Summary: The Classical Robotics Paradox

In classical robotics architectures, the system is fundamentally fractured across a dangerous abstraction boundary:

1. **The Geometry Layer (SBC / Nav2):** Treats the robot as a massless, frictionless point in $\mathbb{R}^2$. It plans kinematics $(x, y, \theta, v, \omega)$ with zero awareness of motor heating, battery voltage sag, chassis inertia, or ground friction.
2. **The Actuation Layer (MCU / Inverter):** Operates as a set of isolated dyno test benches. It tunes high-gain PIDs to violently reject any physical interaction with the environment.
3. **The Illusory Contract (`JointState: position, velocity, effort`):** The high-level controller attempts to calculate `effort` ($\tau = m \cdot a + \dots$) using hardcoded, static parameters. However, because the SBC has no real-time telemetry of load changes, tire scrub variations, or surface friction $\mu$, this computed `effort` is physically decoupled from reality.

In the **Era of Physical AI (Embodied Intelligence)**, a mobile robot is no longer treated as a camera glued to industrial servos. It is an **integrated electro-mechanical organism** where the high-level perception/planning brain and the low-level motor reflex share a **living physical digital twin**.

---

## 2. The Physical AI Two-Tier Hierarchy: Cerebrum vs. Brainstem

In biological systems, the cerebral cortex does not send raw motor unit millivolts to muscle fibers. It sends **force-motion intent** ($\theta, \dot{\theta}, \tau_{ff}$), while the spinal cord and muscle spindles execute **high-bandwidth reflexive impedance** ($1\text{ kHz}$) to prevent falls and tissue damage.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   SINGLE SOURCE OF TRUTH (LIVING DIGITAL TWIN)           │
│       Unified Physical Model (URDF / SDF / Live Parameter Server)        │
│          M(t) [Mass], Izz [Inertia], b [Track], r [Radius], μ, Kt        │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
┌─────────────────────────────────────┐ ┌──────────────────────────────────┐
│ THE CEREBRUM (SBC / Physical AI)    │ │ THE BRAINSTEM (MCU / High-Rate)  │
│ (20–100 Hz / Macro-Physics)         │ │ (100–1000 Hz / Micro-Physics)    │
│                                     │ │                                  │
│ • Perception & Terrain Prediction   │ │ • Sub-Millisecond Muscle Reflex  │
│   (Estimates surface friction μ)    │ │ • FOC Current Vector Control     │
│ • Clothoid G² Geometric S-Curves    │ │ • Virtual Impedance Damping      │
│ • Dynamic Feedforward Force (Effort)│ │   (Prevents slip & lift runaway) │
│ • Online Adaptive Mass Learning     │ │ • Haptic Collision Sensing       │
│   (Recursive Least Squares / PINN)  │ │ • Compliant Docking Mode         │
└──────────────────┬──────────────────┘ └──────────────────┬───────────────┘
                   │                                       │
                   │   THE UNIFIED CONTRACT (JointState)   │
                   │   θ_des(t),  ω_des(t),  τ_ff(t)       │
                   └───────────────────►───────────────────┘
```

---

## 3. The Unified Control Contract (Feedforward Effort + Local Impedance)

What does sending `(position, velocity, effort)` actually mean in Physical AI?

It is **NOT** a command for the MCU to blindly obey one and ignore the others. It is a **Generalized Impedance Control Contract**:

$$\tau_{\text{motor}}(t) = \underbrace{\tau_{\text{ff}}(t)}_{\text{SBC Predictive Muscle}} + \underbrace{K_p \cdot \big(\theta_{\text{des}}(t) - \theta_{\text{act}}(t)\big)}_{\text{Spatial Centering Spring}} + \underbrace{B_d \cdot \big(\dot{\theta}_{\text{des}}(t) - \dot{\theta}_{\text{act}}(t)\big)}_{\text{High-Rate Viscous Damper}}$$

### Physical Operation Modes:
1. **Nominal Tracking (Model Matches Reality):**
   * $\theta_{\text{act}} = \theta_{\text{des}}$ and $\dot{\theta}_{\text{act}} = \dot{\theta}_{\text{des}}$.
   * The feedback errors are **zero**.
   * The robot glides along clothoid spirals powered **100% by the SBC's predictive dynamic torque ($\tau_{\text{ff}}$)** with zero tracking delay.
2. **Terrain Disturbances & Slip (Lift / Ice / Mud):**
   * If a wheel leaves the ground (the "Effective Mass Collapse"), the local damper $B_d (\dot{\theta}_{\text{des}} - \dot{\theta}_{\text{act}})$ injects immediate negative braking torque in $< 1\text{ ms}$, holding wheel speed at $\dot{\theta}_{\text{des}}$ without running away.
3. **Compliant Docking & Contact:**
   * When contacting a charging station, the controller yields compliantly based on $K_p$ and $B_d$, smoothly absorbing mechanical misalignments without bending dock pins.

---

## 4. Resolving the "Ignorant SBC" Problem: Online Dynamic Identification

How does the SBC know the real dynamics if reality constantly changes? **The robot learns its own physical parameters online.**

The MCU streams high-rate telemetry back to the SBC:
$$\mathbf{y}(t) = \big[\tau_{\text{meas}}(t),\; a_{\text{imu}}(t),\; \omega_{\text{gyro}}(t),\; \dot{\theta}_{\text{wheels}}(t)\big]^T$$

The SBC runs an online **Physics-Informed Estimator (Recursive Least Squares or EKF)**:

$$\tau_{\text{meas}} = M_{\text{eff}} \cdot r \cdot a + c_{\text{visc}} \cdot v + \tau_{\text{scrub}} \cdot \tanh(\omega / \varepsilon)$$

```
               [MCU High-Rate Telemetry: Iq, a_imu, omega_gyro]
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ RESIDUAL TORQUE OBSERVER                                                 │
│   e_tau = tau_meas - (M_model · r · a + c_visc · v + tau_scrub)         │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
   Consistent e_tau during                           Sudden e_tau spike
       acceleration:                                     at v ≈ 0:
            │                                                 │
            ▼                                                 ▼
┌─────────────────────────────────────────┐       ┌────────────────────────┐
│ ADAPTIVE MASS LEARNING                  │       │ COLLISION / DOCK HIT   │
│ Update M_eff = M_0 + ΔM                 │       │ Trigger Soft Latch     │
│ (Detected 50 kg cargo addition)         │       │ or E-Stop              │
│ Publishes to ROS 2 Parameter Server     │       └────────────────────────┘
└─────────────────────────────────────────┘
```

1. **Mass Identification:** When accelerating, if current demand consistently exceeds $M \cdot a$, the estimator identifies added payload ($M \leftarrow M + \Delta M$) and updates the ROS 2 Parameter Server. Both SBC and MCU adapt instantly.
2. **Tire Scrub & Ground Identification:** On turns, if the yaw moment fails to achieve expected angular acceleration, the estimator updates `yaw_scrub_torque_nm` for that specific ground surface.

---

## 5. Firmware Implementation Roadmap (MCB G6)

| Layer | Component | Function | Status |
| :--- | :--- | :--- | :--- |
| **MCU** | `drive.c` | Fixed command timeout latching (`valid_ms`); clean transition to `DRIVE_STATE_STOPPED`. | **Completed (`a1e59e2`)** |
| **MCU** | `drive_2dof.c` | Gated breakaway stiction feedforward; per-wheel impedance damping to prevent lift runaway. | **Completed (`a1e59e2`)** |
| **MCU** | `xparams.c` | Dynamic parameter server for live tuning (`mass_kg`, `kp_v`, `yaw_scrub_torque_nm`). | **Completed (`e3c3260`)** |
| **SBC** | `clothoid_fitter` | $G^2$ Clothoid spatial curve fitting from discrete waypoints. | **Operational** |
| **Bridge** | `pico-ROS` | Streaming 200 Hz `sensor_msgs/JointState` ($\theta, \dot{\theta}, \tau_{\text{ff}}$). | **Specified in ARCHITECTURE.md** |
