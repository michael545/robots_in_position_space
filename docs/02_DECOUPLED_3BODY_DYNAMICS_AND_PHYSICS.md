# Document 02: Decoupled 3-Body Dynamics & Continuous Friction Modeling

## 1. The Multi-Body System Definition

A differential drive mobile robot cannot be accurately modeled as a 1-dimensional point mass. It is a coupled multi-body mechanical system consisting of **three distinct rigid bodies**:
1. **Body 1: The Main Chassis**
   - Mass: $M$ [kg]
   - Yaw Moment of Inertia about vertical center of mass axis: $I_z$ [$\text{kg}\cdot\text{m}^2$]
2. **Body 2: Left Wheel & Hub Motor Assembly**
   - Mass: $m_w$ [kg]
   - Rolling Inertia about axle shaft: $I_w$ [$\text{kg}\cdot\text{m}^2$]
   - Lateral offset from center of mass: $-b/2$ [m]
3. **Body 3: Right Wheel & Hub Motor Assembly**
   - Mass: $m_w$ [kg]
   - Rolling Inertia about axle shaft: $I_w$ [$\text{kg}\cdot\text{m}^2$]
   - Lateral offset from center of mass: $+b/2$ [m]

```text
===================================================================================================
                             THE 3-BODY PHYSICAL SCHEMATIC
===================================================================================================
                                      ▲ X_body (Forward / Longitudinal)
                                      │
                         Left Wheel   │          Right Wheel
                         (mw, Iw)     │          (mw, Iw)
                       ┌───────────┐  │        ┌───────────┐
                       │           │  │        │           │
                       │     X     │──┼────────│     X     │
                       │           │  │        │           │
                       └───────────┘  │        └───────────┘
                             ▲        │              ▲
                             │        │              │
                             └────────┼──────────────┘
                             ◄─────── b ─────────────►
                                      │
                         ◄────────────┴──────────────── Y_body (Lateral)
                                  Chassis (M, Iz)
===================================================================================================
```

---

## 2. Derivation of Equivalent Dynamic Parameters via Kinetic Energy

When the robot moves, kinetic energy is stored in both linear translational momentum and rotational angular momentum.

### 2.1 Derivation of Effective Mass ($M_{eff}$)
Consider pure forward translation where $v_L = v_R = v$, and wheel rotational velocity is $\omega_{wheel} = \frac{v}{r}$:

$$T_{trans} = \underbrace{\frac{1}{2} M v^2}_{\text{Chassis Translation}} + \underbrace{2 \times \left(\frac{1}{2} m_w v^2\right)}_{\text{2 Wheels Translation}} + \underbrace{2 \times \left(\frac{1}{2} I_w \omega_{wheel}^2\right)}_{\text{2 Motor Rotors Spinning}}$$

Substitute $\omega_{wheel} = \frac{v}{r}$:
$$T_{trans} = \frac{1}{2} M v^2 + m_w v^2 + I_w \left(\frac{v}{r}\right)^2$$
$$T_{trans} = \frac{1}{2} \underbrace{\left[ M + 2m_w + 2\frac{I_w}{r^2} \right]}_{\mathbf{M_{eff}}} v^2$$

$$\mathbf{M_{eff} = M + 2m_w + 2\frac{I_w}{r^2}}$$

#### Physical Meaning:
The term $2\frac{I_w}{r^2}$ is the **reflected rotational inertia** of the motor armatures, magnets, and wheel hubs. Even if the robot frame weighs $40\,\text{kg}$, spinning up the wheel rotors makes the robot dynamically respond as if its mass were $44\,\text{kg}$ to $48\,\text{kg}$.

---

### 2.2 Derivation of Effective Yaw Inertia ($I_{eff}$)
Consider pure rotation in place at angular yaw rate $\omega = \dot{\theta}_{yaw}$.
* The left wheel orbits at radius $-b/2 \implies v_{trans\_L} = -\omega \frac{b}{2}$.
* The right wheel orbits at radius $+b/2 \implies v_{trans\_R} = +\omega \frac{b}{2}$.
* The wheel rolling spin speeds are $\omega_{wheel} = \frac{|v_{trans}|}{r} = \frac{\omega \cdot b}{2r}$.

Summing all rotational kinetic energy:
$$T_{rot} = \underbrace{\frac{1}{2} I_z \omega^2}_{\text{Chassis Yaw}} + \underbrace{2 \times \left[ \frac{1}{2} m_w \left(\omega \frac{b}{2}\right)^2 \right]}_{\text{Wheels Orbiting Track}} + \underbrace{2 \times \left[ \frac{1}{2} I_w \left(\frac{\omega b}{2r}\right)^2 \right]}_{\text{Rotors Spinning on Axles}}$$

Factoring out $\frac{1}{2}\omega^2$:
$$T_{rot} = \frac{1}{2} \underbrace{\left[ I_z + \frac{m_w b^2}{2} + \frac{I_w b^2}{2r^2} \right]}_{\mathbf{I_{eff}}} \omega^2$$

$$\mathbf{I_{eff} = I_z + \frac{m_w b^2}{2} + \frac{I_w b^2}{2r^2}}$$

#### Physical Meaning:
The effective yaw inertia is substantially larger than $I_z$ alone. Swinging the physical motors around the track width adds $\frac{m_w b^2}{2}$, and accelerating their spin adds $\frac{I_w b^2}{2r^2}$.

---

## 3. Decoupled Modal Equations of Motion

We decompose the generalized vehicle wrench into two orthogonal modal coordinates:
1. **Longitudinal Translation ($F_{linear}$ in Newtons)**
2. **Lateral-Yaw Rotation ($\tau_{rotational}$ in Newton-meters)**

### 3.1 Channel 1: Longitudinal Force Balance
$$F_{linear}(t) = M_{eff} \cdot a(t) + F_{friction\_lin}(v) + C_{f\_lin} \cdot v(t) + C_{d\_lin} \cdot v(t)^2 \cdot \text{sgn}(v)$$

* $M_{eff} \cdot a(t)$: Inertial acceleration force ($F = ma$).
* $F_{friction\_lin}(v)$: Non-linear Stribeck stiction and rolling resistance.
* $C_{f\_lin} \cdot v$: Viscous bearing grease and eddy-current damping.
* $C_{d\_lin} \cdot v^2$: Aerodynamic and turbulent gearbox fluid drag.

### 3.2 Channel 2: Lateral-Yaw Torque Balance
$$\tau_{rotational}(t) = I_{eff} \cdot \alpha(t) + \tau_{friction\_rot}(\omega) + C_{f\_rot} \cdot \omega(t) + C_{d\_rot} \cdot \omega(t)^2 \cdot \text{sgn}(\omega)$$

* $I_{eff} \cdot \alpha(t)$: Yaw rotational inertia ($\tau = I\alpha$).
* $\tau_{friction\_rot}(\omega)$: Stiction plus **Tire Scrubbing Torque** opposing rotation.
* $C_{f\_rot} \cdot \omega$: Caster pivot bearing viscous damping.
* $C_{d\_rot} \cdot \omega^2$: High-speed rotational spin drag.

---

## 4. The Continuous Stribeck Friction Tribology (Zero Infinite Jerk)

### 4.1 The Disastrous Discontinuity of Textbook Friction
Traditional friction models write:
$$F = \begin{cases} C_s \cdot \text{sgn}(v) & |v| \approx 0 \\ C_r \cdot \text{sgn}(v) + C_f v & |v| > 0 \end{cases}$$

Because the sign function has a vertical step at $v = 0$:
$$\frac{dF}{dv}\Big|_{v=0} = \infty \implies \text{Jerk} = \frac{1}{m}\frac{dF}{dt} = \frac{1}{m}\left(\frac{dF}{dv} \cdot a\right) = \mathbf{\infty}$$
When encoder velocity jitters between $\pm 0.001\,\text{m/s}$ at standstill, the motor current slams between $+C_s$ and $-C_s$ at 1 kHz, causing loud audible motor buzzing and gear tooth pitting.

```text
===================================================================================================
                    CONTINUOUS STRIBECK PROFILE (SMOOTH & BOUNDED)
===================================================================================================

  Friction Force F
        ▲
Cs + Cr ┼ - - - - - - - .  ◄── Stiction breakaway peak (no step jump!)
        │              / \
     Cr ┼ - - - - - - / - - \─────────────────────────────────────── ◄── Coulomb rolling resistance
        │            /       '--..__________________..--''  ◄── Viscous linear slope (Cf * v)
        │           /
        │          /  ◄── Continuous S-curve transition across zero! (Slope <= (Cs+Cr)/eps)
        │         /       dF/dt is strictly FINITE -> ZERO JERK SPIKES!
      0 ┼────────┴────────────────────────────────────────────────► Velocity v
        │        vs
===================================================================================================
```

### 4.2 The Mathematical Formula
We eliminate the discontinuity by modeling the physical Stribeck effect using a smooth exponential decay and a hyperbolic tangent:

$$\mathbf{F_{friction}(v) = \left[ C_r + C_s \cdot e^{-\left(\frac{|v|}{v_s}\right)^2} \right] \cdot \tanh\left(\frac{v}{\epsilon}\right) + C_f \cdot v}$$

1. **Breakaway Zone ($|v| \to 0$):**
   - $e^0 = 1 \implies$ Friction reaches full breakaway stiction: $C_s + C_r$.
   - The motor commands instant breakaway force on tick zero, eliminating the startup dead-time.
2. **Decay to Cruising ($|v| \gg v_s$):**
   - The exponential decays to 0, smoothly dropping force to pure Coulomb rolling resistance $C_r$.
3. **Zero-Crossing Jerk Bound:**
   - $\tanh(v/\epsilon)$ replaces $\text{sgn}(v)$.
   - Its derivative is bounded: $\frac{d}{dv}\tanh(v/\epsilon) \le \frac{1}{\epsilon}$.
   - The maximum rate of change of force with time is strictly bounded:
     $$\left|\frac{dF}{dt}\right|_{max} \le \frac{C_s + C_r}{\epsilon} \cdot |a_{max}| < \infty$$
   - Setting $\epsilon = 0.005\,\text{m/s}$ (5 mm/s) provides immediate response without current chatter.

---

## 5. Tire Scrubbing Mechanics on Differential Drive Arcs

When a differential drive robot drives along an arc or rotates in place, the non-steered drive tires and passive casters do not have Ackermann geometry. 

To turn, rubber must physically **scrub laterally across the floor**:
$$\tau_{scrub}(\omega) = C_{r\_rot} \cdot \tanh\left(\frac{\omega}{\epsilon_\omega}\right)$$

* On straightaways ($\omega = 0$), scrub torque is exactly 0.
* On curves ($\omega \neq 0$), scrub torque immediately demands an opposing yaw moment.
* By computing $\tau_{scrub}$ in feedforward, the outside wheel receives extra forward thrust and the inside wheel receives braking torque, **preventing the robot from bogging down in curve apexes**.

---

## 6. Production C Implementation (`drive_physics.c`)

```c
#include "drive_physics.h"
#include <math.h>

static inline float fast_tanh(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

void drive_physics_calculate(
    const drive_physics_params_t* p,
    float v_cmd, float a_lin,
    float w_cmd, float alpha_rot,
    drive_feedforward_output_t* out)
{
    const float EPSILON_V = 0.005f;  // 5 mm/s velocity boundary
    const float EPSILON_W = 0.010f;  // 10 mrad/s yaw boundary

    // 1. Longitudinal Force Calculation
    float f_inertia = p->mass_eff_kg * a_lin;
    float v_ratio = v_cmd / p->vs_lin_m_s;
    float stiction_decay_lin = p->Cs_lin_N * expf(-(v_ratio * v_ratio));
    float direction_lin = fast_tanh(v_cmd / EPSILON_V);
    float f_friction = (p->Cr_lin_N + stiction_decay_lin) * direction_lin;
    float f_viscous = p->Cf_lin_N_s_m * v_cmd;
    float f_drag = p->Cd_lin_N_s2_m2 * (v_cmd * fabsf(v_cmd));

    out->force_linear_n = f_inertia + f_friction + f_viscous + f_drag;

    // 2. Lateral-Yaw Torque Calculation
    float tau_inertia = p->Iz_eff_kg_m2 * alpha_rot;
    float w_ratio = w_cmd / p->vs_rot_rad_s;
    float stiction_decay_rot = p->Cs_rot_Nm * expf(-(w_ratio * w_ratio));
    float direction_rot = fast_tanh(w_cmd / EPSILON_W);
    float tau_friction = (p->Cr_rot_Nm + stiction_decay_rot) * direction_rot;
    float tau_viscous = p->Cf_rot_Nm_s_rad * w_cmd;
    float tau_drag = p->Cd_rot_Nm_s2 * (w_cmd * fabsf(w_cmd));

    out->torque_yaw_nm = tau_inertia + tau_friction + tau_viscous + tau_drag;
}
```
