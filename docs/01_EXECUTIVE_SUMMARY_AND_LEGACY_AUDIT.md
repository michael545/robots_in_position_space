# Document 01: Executive Summary & Legacy Control Audit

## 1. Architectural Overview & Context

This document captures the systems engineering analysis of the mobile robot motion control stack on the **Ubiquity Robotics Magni MCB G6 platform** (STM32G474QETX microcontroller + Raspberry Pi 5 / Robinson's Lake Single Board Computer).

The platform is transitioning from a legacy reactive velocity-space paradigm (`cmd_vel`) to a deterministic **Continuous-to-Discrete (C2D) Position-Space Trajectory Architecture**.

```text
===================================================================================================
                               SYSTEM CONTROL DOMAIN TRANSITION
===================================================================================================

 LEGACY VELOCITY PIPELINE (DEPRECATED):
 [Nav2 Planner] ──► [move_smooth (20Hz)] ──► [/cmd_vel (Twist)] ──► [drive_process (100Hz)] ──► [Wheels]
                         ▲ Latency & Jitter      ▲ No spatial data      ▲ Open-loop integration drift

 REPLACEMENT C2D POSITION PIPELINE:
 [Nav2 Planner] ──► [Clothoid Fitter (G²)] ──► [Ruckig S-Curve] ──► [200Hz JointState] ──► [1kHz FOC]
                         ▲ Continuous curvature   ▲ Jerk-bounded time    ▲ Pos + Vel-FF + Torque-FF
===================================================================================================
```

---

## 2. The Legacy Philosophy: "Implicit Distance via Dead-Man Time"

The foundation of the legacy ROS 1/2 navigation architecture is captured in the classical Ubiquity Robotics slide:

> *"cmd_vel gives us speed, and defines a 'dead man' time. Distance is thus implicitly defined."*

```text
===================================================================================================
                      THE SLIDE'S THEORETICAL "IMPLICIT DISTANCE" MODEL
===================================================================================================

   Position x(t)
        ▲                                                Flat (Robot stopped)
        │                                         ┌───────────────────────────►
        │                                        ╱ ◄── Distance implicitly defined
        │                                       ╱      D = v * t_dead_man
        │                                      ╱
        │                                     ╱ (Constant slope = v)
        │                       Flat         ╱
        └───────────────────────────────────┴─────────────────────────────────► Time t
                                            ▲     ▲
                                            │     │
                                            │     └─────── "Dead man time" expires
                                            └───────────── "Cmd_vel message issued"

   Velocity v(t)
        ▲
        │                     Rectangular Pulse (Infinite Acceleration!)
        │                        ┌────────────────────────┐
      v ┼────────────────────────┤                        │
        │                        │                        │
        │                        │  Area = Distance       │
        │                        │  D = v * t_dead_man    │
      0 ┴────────────────────────┴────────────────────────┴───────────────────► Time t
                                 ▲                        ▲
                                 │                        │
                      Cmd_vel Issued                  Dead Man Time
                      (Instant jump 0 -> v)          (Instant drop v -> 0)
===================================================================================================
```

### The Theoretical Premise
In this abstraction:
1. The planner transmits instantaneous velocities $\mathbf{v} = (v, \omega)$.
2. The message carries an implicit validity duration $t_{dead\_man}$ (the watchdog timeout, e.g. 300 ms).
3. The planner assumes the robot travels an exact rectangular distance:
   $$D_{implicit} = \int_0^{t_{dead\_man}} v(t)\, dt = v \cdot t_{dead\_man}$$

---

## 3. The Firmware Reality: Deep Audit of `drive.c`

Inspecting the actual microcontroller firmware in [`mcb_g6_firmware/src/modules/drive/drive.c`](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive.c) reveals why the theoretical model breaks down in physical hardware.

### 3.1 `current_speed` is Fictional (Software Simulation)
In [`drive.c` lines 418–447](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive.c#L418-L447):

```c
// Motor acceleration - deceleration loop in drive.c
for (int i = 0; i < DRIVE_N_MOTORS; i++) {
    if (drv->state.target_speed_mm_s[i] >= 0 && drv->state.current_speed_mm_s[i] < drv->state.target_speed_mm_s[i]) {
        drv->state.current_speed_mm_s[i] += accel_step; // Pure software accumulation!
    }
}
```

* **The Code Reality:** `current_speed_mm_s` is **not measured from wheel encoders**. It is an internal mathematical variable that increments by `accel_step` ($a \cdot \Delta t$) every 10 ms.
* If the robot were tied to a post with wheels completely locked, `current_speed_mm_s` would still ramp up to 1000 mm/s in software.

### 3.2 Deceleration Run-Out (Violating the Dead-Man Time)
In [`drive.c` lines 324–329](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive.c#L324-L329):
```c
if (drv->state.last_cmd.valid_ms > 0) {
    if (timer_get_time_ms(drv->state.last_cmd_timer) >= drv->state.last_cmd.valid_ms) {
        drv->state.target_speed_mm_s[DRIVE_M1] = 0;
        drv->state.target_speed_mm_s[DRIVE_M2] = 0;
    }
}
```
* When the dead-man timer expires (`timer >= valid_ms`), the robot **does not stop**.
* Instead, `target_speed` drops to zero, and the firmware begins **deceleration ramping**:
  $$v[k] = v[k-1] - d_{max} \cdot \Delta t$$
* The robot accumulates a significant **run-out stopping distance**:
  $$D_{actual} = v \cdot t_{dead\_man} + \frac{1}{2}\frac{v^2}{d_{max}} - \frac{1}{2}\frac{v^2}{a_{max}}$$
* The distance traveled is therefore variable and dependent on initial speed, battery voltage, and deceleration limits.

### 3.3 Odometry is Completely Ignored on Turns & Transients
Looking at [`drive.c` lines 453–465](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive.c#L453-L465):
```c
if (cruising && drv->state.drive_mode == DRIVE_MODE_STRAIGHT) {
    int16_t pid_out = PID_Controller(&drv->state.pid_follow,
            drv->state.local_frame.y_mm * 100 * drv->state.drive_dir * rw_fwd);
    correction = (float) pid_out / 100.0f;
}
```
* **Odometry never corrects forward speed.** There is no forward speed feedback loop anywhere in `drive.c`.
* The only use of odometry is `local_frame.y_mm`—a lateral offset trim that **only runs when driving straight and cruising**.
* If the robot is in a turn (`DRIVE_MODE_CURVE` or `DRIVE_MODE_ROTATE`), **odometry is 100% ignored**.

### 3.4 Open-Loop Integration into Position Servoing
In [`drive.c` lines 474–476 and 53–56](file:///home/michael/code/firmware/robots_in_position_space/mcb_g6_firmware/src/modules/drive/drive.c#L474-L476):
```c
drv->state.target_rad[DRIVE_M1] += (drv->state.corrected_speed_mm_s[DRIVE_M1] * dt) / w_radius;
drv->state.target_rad[DRIVE_M2] += (drv->state.corrected_speed_mm_s[DRIVE_M2] * dt) / w_radius;
drive_set_motor_targets(drv);
...
MC_ProgramPositionCommandMotor1(drv->state.target_rad[DRIVE_M1], 0);
MC_ProgramPositionCommandMotor2(drv->state.target_rad[DRIVE_M2], 0);
```
The firmware integrates fictional speed into radian targets, passing them to the ST Motor Control SDK (MCSDK) position controller.

---

## 4. The "Dead-Time" Phenomenon: Why Pure Feedback Position Servoing Fails

The second Ubiquity Robotics slide diagnoses the core physical bottleneck:

> *"After the cmd_vel message is issued, the controller has to wait until the motor is out of position to generate any force. This takes time. Position based control results in a less responsive controller."*

```text
===================================================================================================
                     THE "DEAD-TIME" AND OVERSHOOT PHENOMENON
===================================================================================================

 1. POSITION x(t): The Tracking Delay
    x ^                       Desired Target (target_rad accumulating)
      │                  _.-' - - - - - - - - - - - - - - - - - - - - - - - - -
      │             _.-' _.-'
      │        _.-'   .-'   ◄── Actual Motor Position (Lags behind by tracking error e)
      │   _.-'     .-'
      │  '_______.-'  ◄── "Dead time": Motor sitting still waiting for e * Kp > stiction
      +───────────────────────────────────────────────────────────► Time t
                 ▲
                 └─ Cmd_vel issued

 2. VELOCITY v(t): The Overshoot Hump
    v ^       ┌ - - - - - - - - - - - - - - - - - - Desired Speed Target
      │       │       _.._  ◄── Overshoot! (Motor snaps forward to catch up)
      │       │     .'    '.
      │       │    /        `'--..____  ◄── Settles into steady-state lag
      │- - - -+.../───────────────────
      +───────────────────────────────────────────────────────────► Time t
              ▲
              └─ Lag before motor actually begins moving
===================================================================================================
```

### The Physics of the Dead Time:
1. In ST MCSDK, motor torque is proportional to position error:
   $$\tau = K_p \cdot (\theta_{target} - \theta_{actual})$$
2. At $t = 0$, $\theta_{target} \approx \theta_{actual} = 0 \implies \mathbf{\tau = 0}$.
3. The motor produces **zero initial torque**.
4. The motor cannot move until $\theta_{target}$ crawls ahead enough that:
   $$K_p \cdot e > \tau_{stiction} + \tau_{inertia}$$
5. **The Spring Snap:** While the motor is frozen on static friction, error accumulates like a stretched mechanical spring. When breakaway occurs, the stored error releases violently, causing an **overshoot hump in velocity** and chassis jerk.

---

## 5. Architectural Summary of Legacy Deficiencies

| Deficiency | Mechanism in Legacy Code | Physical Manifestation |
| :--- | :--- | :--- |
| **Startup Dead-Time** | Pure feedback position servoing ($K_p \cdot e$) | Robot sits stationary for 20–80 ms after command arrival |
| **Velocity Overshoot** | Spring-snap release of accumulated position error | Violent jerk and wheel slip immediately after breakaway |
| **Corner Understeer** | No yaw moment feedforward ($\tau_{yaw} = 0$) | Chassis pushes straight forward into curve, swinging wide |
| **Apex Speed Sag** | No tire scrub friction compensation | Robot bogs down and loses 20–30% speed in curve apex |
| **Odometry Corruption** | Reactive torque spikes cause wheel skidding | Wheel encoders report false distance; localizer drifts |
| **Timing Non-Determinism** | Open-loop integration over variable $\Delta t_{arrival}$ | Speed and position commands drift with DDS/serial jitter |
