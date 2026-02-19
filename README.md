# Robot Control in Position Space: Architectural Analysis

This document synthesizes the architectural transition from velocity-based control (`cmd_vel`) to position-space trajectory tracking for autonomous ground vehicles (AGVs). It details the control theory, the bottlenecks in standard ROS 2 navigation stacks, and the implementation of deterministic hardware interfaces.

## 1. The Velocity Bottleneck

Standard ROS 2 navigation stacks (Nav2) typically operate on a "Sense-Plan-Act" cycle that outputs a velocity command (`geometry_msgs/Twist`) at 20Hz-50Hz. This introduces a fundamental control problem when aiming for high-precision, smooth motion.

### The Problem: Redundant Integration
1.  **Planner (MPPI/TEB):** Calculates a precise trajectory $(x(t), y(t), 	heta(t))$ and its derivatives $(\dot{x}, \dot{y}, \dot{	heta})$.
2.  **Controller Output:** Truncates this trajectory to a single instantaneous velocity vector $\vec{v}_{cmd}$ (Linear $v$, Angular $\omega$).
3.  **Hardware Interface:** Transmits $\vec{v}_{cmd}$ to the firmware.
4.  **Firmware Loop:** Re-integrates velocity to estimate a target position:
    $$ 	heta_{target}(t+\Delta t) = 	heta_{target}(t) + \dot{	heta}_{measured} \cdot \Delta t $$

**Consequences:**
*   **Loss of Future Intent:** The firmware only knows the *instantaneous* velocity, not where the robot needs to be in 500ms. It cannot pre-compensate for upcoming curves.
*   **Drift:** Any error in $\dot{	heta}_{measured}$ (sensor noise, wheel slip) is permanently integrated into the position estimate. The robot believes it is at the correct position when it has actually drifted.
*   **Latency:** The "Stop-and-Go" nature of 20Hz velocity updates causes micro-accelerations and decelerations, leading to jerky motion and mechanical wear.

## 2. The Solution: Trajectory Tracking

To achieve industrial-grade performance, we must bypass the velocity bottleneck and command the robot in **Joint Space** using **Trajectory Points**.

### 2.1 The Data Structure (`JointTrajectoryPoint`)
A trajectory point defines the complete kinematic state of a joint at a specific time $t$:

*   **Position ($q$):** The primary control target.
*   **Velocity ($\dot{q}$):** The feedforward term (Feedforward Velocity).
*   **Acceleration ($\ddot{q}$):** The inertial compensation term (Feedforward Torque).
*   **Effort ($	au$):** Force/Torque feedforward (gravity/friction compensation).

### 2.2 Control Theory: Spline Interpolation
The `joint_trajectory_controller` (JTC) connects discrete trajectory points using **Quintic Splines**. A quintic spline is a 5th-order polynomial $q(t)$ that ensures continuity in position, velocity, and acceleration.

Given two points:
*   $P_0$ at $t_0$: $(q_0, \dot{q}_0, \ddot{q}_0)$
*   $P_1$ at $t_1$: $(q_1, \dot{q}_1, \ddot{q}_1)$

The controller solves for coefficients $a_0 \dots a_5$ such that:
$$ q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5 $$

This guarantees:
1.  $q(t_0) = q_0$ (Hits the start point)
2.  $\dot{q}(t_0) = \dot{q}_0$ (Matches start velocity)
3.  $q(t_1) = q_1$ (Hits the end point)
4.  $\dot{q}(t_1) = \dot{q}_1$ (Matches end velocity)

**Why this matters:**
By respecting the velocity constraints at the boundaries, the robot's motion is fluid. The end velocity of Segment A becomes the start velocity of Segment B, eliminating infinite acceleration spikes (jerk) at waypoints.

### 2.3 Firmware Implementation: Feedforward Control
The firmware receives the interpolated setpoint $(q_{target}, \dot{q}_{target})$ from JTC at a high rate (e.g., 100Hz-1kHz).

The control law is a **Feedforward + Feedback** loop:

$$ u(t) = K_p (q_{target} - q_{measured}) + K_v (\dot{q}_{target}) + K_a (\ddot{q}_{target}) $$

*   **$K_p (q_{err})$:** Corrects for disturbances (bumps, friction).
*   **$K_v (\dot{q}_{target})$:** The dominant term. It provides the base voltage required to maintain the commanded velocity.
*   **$K_a (\ddot{q}_{target})$:** Inject extra current during acceleration phases to overcome rotor inertia.

This structure allows the PID gains ($K_p$) to be lower (softer), reducing oscillation, because the Feedforward terms are doing 90% of the work.

## 3. Architecture for Nav2 & MPPI

To implement this with MPPI (Model Predictive Path Integral):

1.  **Global Planner:** Generates the route.
2.  **MPPI Controller:**
    *   Optimizes a trajectory in Task Space $(x, y, 	heta)$.
    *   **Crucial Step:** Instead of truncating to `cmd_vel`, extract the optimal trajectory (e.g., top 1-2 seconds).
3.  **Inverse Kinematics (IK) Node:**
    *   Converts Task Space Trajectory $ightarrow$ Joint Space Trajectory.
    *   For Differential Drive:
        $$ \Delta q_L = \Delta s - (\Delta 	heta \cdot \frac{L}{2}) $$
        $$ \Delta q_R = \Delta s + (\Delta 	heta \cdot \frac{L}{2}) $$
        (Where $\Delta s$ is arc length and $L$ is wheel base).
4.  **Hardware Interface:**
    *   Accepts `JointTrajectory` messages.
    *   Sends `(Position, Velocity)` packets to firmware.

## 4. Safety Considerations

Operating in Position Space introduces specific risks:

*   **Integral Windup:** If the robot is physically blocked, the position error grows indefinitely.
    *   *Mitigation:* Clamping the integrator term and implementing "Following Error" limits.
*   **Communication Loss:** If the stream of points stops, the robot will try to hold the last position with full torque.
    *   *Mitigation:* Firmware Watchdog. If no packet in $X$ ms, set velocity/torque to 0.
*   **Startup Jumps:** If the controller starts with a setpoint far from the current encoder reading, the robot will jump violently.
    *   *Mitigation:* "bumpless transfer" - initialize the setpoint to the current encoder value upon enabling.

## 5. Firmware Protocol Definition

To support this architecture, the firmware communication protocol typically uses a struct like:

```c
typedef struct {
    float pos[N_MOTORS];       // Target Angle (Radians)
    float vel[N_MOTORS];       // Target Velocity (Radians/s) - Feedforward
    float effort[N_MOTORS];    // Target Torque (Nm) - Feedforward
    uint32_t valid_ms;         // Watchdog timeout
} drive_joint_cmd_t;
```

This packet allows the firmware to execute the full Feedforward control law described in Section 2.3.
