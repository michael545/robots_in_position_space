# Document 09: Outdoor Terrain Dynamics, Incline Physics & Gravity Compensation

## 1. Executive Summary: The Outdoor Robotics Dilemma

A common and critical engineering skepticism arises when transitioning from laboratory flat floors to harsh, dynamic outdoor environments:

> *"If surface friction ($\mu$) changes constantly, if ground slopes change continuously, if starting inclines vary, and if tire scrub differs between mud, gravel, and grass—what is the point of implementing a dynamics model in firmware? Doesn't outdoor terrain invalidate static models?"*

The answer from first-principles control engineering is: **Outdoor terrain does not invalidate firmware dynamics—it is the very reason why firmware dynamics are mandatory.**

Without a physical dynamics model in firmware, an outdoor autonomous mobile robot (AMR) suffers from three catastrophic failures:
1. **Incline Rollback:** When attempting to start on a $15^\circ$ slope, a position-only controller rolls backward by several centimeters before generating enough restoring torque, endangering people and equipment.
2. **Hill Runaway / Speed Sag:** Going uphill, a naive controller lags severely; going downhill, it accelerates uncontrollably until feedback saturates.
3. **Mud & Gravel Slip-Spike (Trenching):** When an outdoor wheel transitions from firm soil to mud, the effective inertia collapses ($80\times$), causing the wheel to instantly spin out and dig itself into a trench before the high-level SBC can even receive the odometry packet.

This document formalizes how the **Magni MCB G6 firmware** pairs the **Physical Invariants** with **Real-Time Onboard IMU Sensor Telemetry** to conquer dynamic outdoor terrain.

---

## 2. The Physical Invariants vs. Terrain Variables

To understand why firmware dynamics are essential, we must separate the robot's physics into two distinct mathematical categories:

```text
===================================================================================================
                     PHYSICAL INVARIANTS vs. TERRAIN DYNAMIC VARIABLES
===================================================================================================

  CATEGORY A: PHYSICAL INVARIANTS (True Everywhere in the Universe)
  -------------------------------------------------------------------------------------------------
  • Vehicle Mass:             M = 100 kg         (Does NOT change when driving onto mud)
  • Rotational Yaw Inertia:   I_zz ≈ 5.3 kg·m²   (Does NOT change when climbing a slope)
  • Wheel Radius:             r = 0.200 m        (Geometric constant)
  • Track Width:              b = 0.655 m        (Geometric constant)
  • Motor Torque Constant:    K_t = 4.8 Nm/A     (Electromagnetic invariant of windings & magnets)
  • Newton's Second Law:      F = M · a          (Force required to accelerate mass is invariant)

  CATEGORY B: TERRAIN DYNAMIC VARIABLES (Vary Continuously Outdoors)
  -------------------------------------------------------------------------------------------------
  • Incline Grade (Pitch):    θ_pitch            (Measured live by onboard IMU: mcb.inav)
  • Side-Slope Bank (Roll):   θ_roll             (Measured live by onboard IMU: mcb.inav)
  • Surface Friction:         μ ∈ [0.2, 0.9]     (Wet grass, mud, loose gravel, dry asphalt)
  • Skid-Steer Scrub Torque:  τ_scrub            (Low in mud ~15 Nm, High on dry asphalt ~55 Nm)
  • Axle Normal Loads:        N_front, N_rear    (Weight transfers dynamically on slopes & accel)
===================================================================================================
```

**The Core Takeaway:**
The firmware does **not** hardcode the terrain variables—it implements the **Physical Invariants ($M, I_{zz}, r, b, K_t$)** and fuses them in real-time ($100\text{--}1000\text{ Hz}$) with the **onboard sensors (`mcb.inav` IMU, phase current $I_q$, and high-speed timers)** to instantaneously solve the Terrain Variables.

---

## 3. Incline Physics & Gravity Force Balance

When an outdoor AMR navigates an incline with angle $\theta_{\text{pitch}}$, the earth's gravitational acceleration vector $\mathbf{g} = [0, 0, -9.81\,\text{m/s}^2]^T$ projects directly onto the vehicle's body axes:

```text
                           Slope Angle θ
                            ▲
                           /
                          /      Chassis (M = 100 kg)
                         /     ┌───────────────────┐
                        /      │       C.G.        │
                       /       │         ●         │──► F_thrust (Motors)
                      /        └───────────────────┘
                     /             ▲           │
                    /              │ N         │ F_gravity = M · g
                   /               │           ▼
                  /           ──────────────────────────────────
                 /            F_slope = M · g · sin(θ) (Resisting Force)
                /             N_normal = M · g · cos(θ) (Tractive Normal Force)
               /
  ────────────┴────────────────────────────────────────────────►
```

### 3.1 Longitudinal Force Balance on Slopes
$$\Sigma F_x = F_{\text{thrust}} - F_{\text{gravity}} - F_{\text{rolling}} = M_{\text{eff}} \cdot a_{\text{des}}$$

Where:
$$F_{\text{gravity}} = M \cdot g \cdot \sin(\theta_{\text{pitch}})$$
$$N_{\text{normal}} = M \cdot g \cdot \cos(\theta_{\text{pitch}})$$

For a $100\text{ kg}$ robot on a $15^\circ$ hill ($\sin(15^\circ) \approx 0.259$):
$$F_{\text{gravity}} = 100 \times 9.81 \times 0.2588 = \mathbf{253.9\,\text{N}} \quad (\approx 25.9\,\text{kgf})$$

To simply stand still or drive forward at a steady $0.5\text{ m/s}$ ($a = 0$), the motors must continuously output:
$$\tau_{\text{incline}} = \frac{1}{2} \cdot r \cdot F_{\text{gravity}} = 0.5 \times 0.200 \times 253.9 = \mathbf{25.4\,\text{Nm}}$$
At $K_t = 4.8\,\text{Nm/A}$, this requires **$5.3\,\text{A}$ of continuous $I_q$ current** just to balance gravity!

### 3.2 Why Firmware Must Compute This: Hill-Start Anti-Rollback
* **In a Blind Position Loop:**
  When stopped on a $15^\circ$ hill, the user commands forward motion. The position setpoint advances. However, at $t = 0$, the error is zero, so the motor outputs zero torque. Gravity ($254\,\text{N}$) immediately accelerates the 100 kg chassis **backward down the hill**. Only after rolling backward by $5\text{--}10\,\text{cm}$ does the position error accumulate enough PID current to reverse direction, causing a violent lurch.
* **With Firmware Gravity Compensation:**
  The MCB G6 firmware reads the true spatial pitch angle directly from the onboard AHRS:
  $$\theta_{\text{pitch}} = \text{mcb.inav.ahrs.euler.angle.pitch}$$
  The firmware computes the gravity feedforward directly:
  $$F_{\text{gravity\_ff}} = M \cdot g \cdot \sin(\theta_{\text{pitch}})$$
  Before the wheels turn, the firmware **pre-loads the motor windings with the exact $25.4\,\text{Nm}$ required to counter gravity**. When motion begins, the AMR moves forward with **zero rollback**.

---

## 4. Outdoor Surface Friction ($\mu$) and Skid-Steer Tire Scrub

In outdoor skid-steer navigation, tire scrub torque varies by over $350\%$ depending on the substrate:

| Substrate | Friction Coefficient ($\mu$) | Turning Scrub Torque ($\tau_{\text{scrub}}$) | Slippage Risk |
| :--- | :---: | :---: | :---: |
| **Dry Concrete / Asphalt** | $0.80\text{--}0.90$ | $\approx 55\,\text{Nm}$ (High resistance) | Very Low |
| **Hard Packed Dirt** | $0.60\text{--}0.70$ | $\approx 35\,\text{Nm}$ (Moderate) | Low |
| **Wet Grass** | $0.35\text{--}0.45$ | $\approx 20\,\text{Nm}$ (Low resistance) | Moderate |
| **Mud / Wet Clay** | $0.20\text{--}0.30$ | $\approx 12\,\text{Nm}$ (Minimal scrub) | Extreme |

### How the Firmware Adapts to Scrub Variations Automatically:
In [`drive_2dof.c`](file:///home/michael/code/firmware/mcb_g6_firmware/src/modules/drive/drive_2dof.c), the yaw steering moment is governed by:
$$\tau_{\text{yaw\_total}} = \tau_{\text{iner\_ff}} + \tau_{\text{scrub\_ff}} + \underbrace{K_{p,w} \cdot (\omega_{\text{des}} - \omega_{\text{gyro}})}_{\text{High-Rate IMU Gyro Feedback}}$$

1. **On High-Friction Asphalt:**
   Tires resist rotation. Without enough torque, the robot will understeer ($\omega_{\text{gyro}} < \omega_{\text{des}}$). The error $(\omega_{\text{des}} - \omega_{\text{gyro}})$ grows positive, and the IMU feedback loop immediately injects additional differential torque ($\Delta \tau$) until the AMR turns at the exact desired rate.
2. **On Low-Friction Mud:**
   Tires turn easily. The feedforward $\tau_{\text{scrub\_ff}}$ would normally cause oversteer. However, the instant the chassis starts rotating faster than commanded ($\omega_{\text{gyro}} > \omega_{\text{des}}$), the error term becomes **negative**, immediately attenuating $\Delta \tau$ and preventing a spin-out!
3. **The Result:**
   The firmware does not need to know the exact ground material in advance. The **closed-loop IMU gyro trims the scrub torque dynamically at 100 Hz**.

---

## 5. 4WD Dynamic Normal Load Redistribution (Weight Transfer)

On a 4WD chassis with two motor control boards (`/mcb1` front axle, `/mcb3` rear axle), climbing a slope redistributes the vertical normal loads across the axles:

```text
       F_front_normal                                F_rear_normal
            ▲                                             ▲
            │                 Slope Angle θ               │
            │                  /                          │
         ┌──┴──┐              /                        ┌──┴──┐
         │Front│             /                         │Rear │
         │Axle │            /                          │Axle │
         └─────┘           /                           └─────┘
```

$$N_{\text{front}} = M \cdot g \left( \frac{L_r}{L} \cos\theta - \frac{h_{\text{cg}}}{L} \sin\theta \right)$$
$$N_{\text{rear}} = M \cdot g \left( \frac{L_f}{L} \cos\theta + \frac{h_{\text{cg}}}{L} \sin\theta \right)$$

* On a $15^\circ$ slope, up to **$70\%$ of the vehicle's total weight shifts onto the rear axle**, while the front axle unloads to $30\%$.
* **The Classical Failure:** If both front and rear axles blindly run independent speed PIDs, the unloaded front tires lose tractive capacity ($\tau_{\max} = \mu \cdot N_{\text{front}} \cdot r$) and spin out, while the loaded rear tires starve for torque.
* **The Firmware Solution:** Because `/mcb1` and `/mcb3` participate on the same dynamic parameter and telemetry bus, the system knows the vehicle pitch angle $\theta$ and can bias torque distribution toward the loaded rear axle, preventing front-wheel spinout while maximizing climbing traction.

---

## 6. The Sub-Millisecond Bandwidth Imperative (Why Firmware MUST Do This)

Why can't the Single Board Computer (SBC) running ROS 2 simply handle all of these dynamics?

Look at the **Physical Latency vs. Event Reaction Times**:

```text
===================================================================================================
                               REAL-TIME RESPONSE BANDWIDTH HIERARCHY
===================================================================================================

  EVENT / HARDWARE LAYER                          TIMESCALE        HANDLED BY
  -------------------------------------------------------------------------------------------------
  • FOC PWM Phase Current Shunt Sampling (Ia, Ib)  31.25 μs (32 kHz) Hardware ADC / STM32 Timers
  • Motor Torque Regulation (Iq PI Loop)           62.50 μs (16 kHz) ST MCSDK Core Interrupt
  • Wheel Slip Onset (Effective Mass Collapse)      3.00 ms          Firmware TCS Watchdog
  • Incline Rollback Impulse                        0.00 ms (Instant)Firmware Gravity Pre-load
  • 2-DoF Modal Dynamics & IMU Gyro Loop           10.00 ms (100 Hz) Firmware drive_2dof.c
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  [ THE SERIAL / ZENOH / PICO-ROS TRANSPORT BARRIER: 15–50 ms Latency with Network Jitter ]
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  • SBC C2D Clothoid Trajectory Evaluation          5.00 ms (200 Hz) SBC C++ Thread
  • Nav2 Local Costmap & Path Optimization        100.00 ms (10 Hz)  SBC ROS 2 Node
  • Terrain Classification (LiDAR / Camera)       200.00 ms (5 Hz)   SBC Neural Network
===================================================================================================
```

If you delegate gravity compensation or wheel slip prevention to the SBC:
* By the time the SBC receives the odometry message indicating wheel slip, runs its node, and sends back a corrective torque command, **$40\text{--}60\,\text{ms}$ have elapsed**.
* In $50\,\text{ms}$, an unloaded wheel spinning at $15\,\text{A}$ accelerates past $1500\,\text{RPM}$, loses all directional stability, and digs a hole in the mud.
* **On the MCU**, the IMU and phase currents are sampled directly via hardware registers with **sub-millisecond latency**. The MCU can quench slip and balance gravity before the wheel moves more than a fraction of a millimeter.

---

## 7. The Unified Architecture: How the Stack Operates in Dynamic Outdoors

```
┌────────────────────────────────────────────────────────────────────────┐
│ SBC (Cerebrum): Nav2 + Clothoid C2D Trajectory Controller              │
│                                                                        │
│ • Long-Horizon Spatial Geometry: Fits smooth Clothoid curves (x, y, θ) │
│ • Terrain Perception: Camera/LiDAR identifies grass, mud, or asphalt   │
│ • Macro Limits: Sets v_max and a_max based on path sightlines          │
│ • Streams: v_des(t), w_des(t) to MCU                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ cmd_vel (or JointState) at 20–100 Hz
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MCU (Brainstem): STM32 MCB G6 Firmware                                 │
│                                                                        │
│ 1. Reads Onboard IMU (mcb.inav) at 100 Hz:                             │
│    • Pitch angle θ_pitch ──► Gravity Compensation: F_g = M*g*sin(θ)    │
│    • Gyroscope Z-axis    ──► Spatial Yaw Trimming:   Kp_w*(w_des - w)  │
│ 2. Reads Dynamics Parameter Server (Live ROS 2 Parameters):            │
│    • Mass (M = 100 kg), Wheel Track (b), Radius (r), Scrub Torque      │
│ 3. Solves 2-DoF Modal Dynamics:                                        │
│    • F_total = M_eff*a_des + F_gravity + F_stiction_gate + F_lin_fb    │
│    • tau_total = I_eff*alpha_des + tau_scrub_ff + tau_yaw_fb           │
│ 4. Kinetic Mixer & Virtual Impedance:                                  │
│    • tau_Left, tau_Right with per-wheel damping to quench slip         │
│ 5. Injects directly into ST MCSDK FOC current loops at 16 kHz          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusion

Implementing the physical dynamics model in firmware is **not redundant** with outdoor terrain variations—it is the **enabling foundation** that makes outdoor autonomy physically possible:

1. **It guarantees Newton's laws are respected** at the actuator level ($F = Ma$), so the robot moves with genuine physical mass and momentum.
2. **It eliminates hill rollback** through zero-latency IMU pitch-angle gravity feedforward.
3. **It handles unpredictable surface friction and tire scrub** through closed-loop onboard IMU gyroscope trimming.
4. **It protects the drive train and tires** by detecting and quenching wheel slip within milliseconds.

By combining the **SBC's macro-geometric clothoid intelligence** with the **MCU's high-rate physical dynamics and sensor reflexes**, the AMR achieves rock-solid, professional outdoor navigation across any terrain.
