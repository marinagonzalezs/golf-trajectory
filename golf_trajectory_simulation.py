"""
Simulation of a golf ball trajectory including gravity, drag,
Magnus force and wind effects, using the explicit Euler method.
"""

from math import cos, sin, sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL PHYSICAL CONSTANTS
# =============================================================================

g = 9.81                     # Gravitational acceleration [m/s^2]
rho0 = 1.225                 # Air density at sea level [kg/m^3]
ball_radius = 21.3 / 1e3     # Golf ball radius [m]
area = pi * ball_radius**2   # Cross-sectional area [m^2]
m = 0.045                    # Golf ball mass [kg]
dt = 0.001                   # Time step [s]

# Default initial conditions
x0 = 0.0
y0 = 0.0
z0 = 0.0
t0 = 0.0
vel0 = 70.0
vel0z = 0.0

# Launch angles
alpha_deg = [0, 15, 30, 45, 60, 75, 90]

# Magnus model parameter
b_magnus = 0.25



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def rpm_to_rad_s(omega_rpm):
    """
    Convert angular velocity from rpm to rad/s
    """
    return omega_rpm * 2.0 * pi / 60.0


def air_density(height, model="constant"):
    """
    Return air density according to the selected model
    """
    if model == "constant":
        return rho0

    if model == "adiabatic":
        alpha_exp = 2.5
        a = 6.5e-3
        T0 = 300.0
        rho = rho0 * (1.0 - a * height / T0) ** alpha_exp
        return rho

    if model == "isothermal":
        rho = rho0 * exp(-height / (8000))
        return rho

    raise ValueError("Unknown density model")


def magnus_acceleration(vx, vy, vz, wx_rad, wy_rad, wz_rad):
    """
    Compute Magnus acceleration per unit mass
    """
    omega_mod = sqrt(wx_rad**2 + wy_rad**2 + wz_rad**2)
    if omega_mod == 0.0:
        a_mag = 0.0
    else:
        a_mag = b_magnus / omega_mod

    ax_m = a_mag * (vz * wy_rad - vy * wz_rad)
    ay_m = a_mag * (vx * wz_rad - vz * wx_rad)
    az_m = a_mag * (vy * wx_rad - vx * wy_rad)

    return ax_m, ay_m, az_m


def drag_force(vx, vy, vz, speed, rho):
    """
    Drag force model:
        F_drag_i = C * rho * A * v * v_i
    """
    if speed <= 14.0:
        c_drag = 0.5
    else:
        c_drag = 7.0/speed

    fdx = c_drag * rho * area * speed * vx
    fdy = c_drag * rho * area * speed * vy
    fdz = c_drag * rho * area * speed * vz

    return fdx, fdy, fdz, c_drag


def wind_force(c_drag, rho, speed, wind):
    """
    Wind contribution
    """
    wind_x, wind_y, wind_z = wind

    fwx = c_drag * rho * area * wind_x * speed
    fwy = c_drag * rho * area * wind_y * speed
    fwz = c_drag * rho * area * wind_z * speed

    return fwx, fwy, fwz


def initial_velocity_from_angle(v0, alpha, v0z=0.0):
    """
    Compute initial velocity components from launch angle
    """
    alpha_rad = np.radians(alpha)
    return v0 * cos(alpha_rad), v0 * sin(alpha_rad), v0z


# =============================================================================
# SIMULATION OF THE TRAJECTORY
# =============================================================================
def simulate_trajectory(
    alpha,
    v0=vel0,
    v0z=vel0z,
    omega_rpm=(0.0, 0.0, 0.0),
    use_drag=True,
    wind=(0.0, 0.0, 0.0),
    density_model="constant",
    dt_local=dt
):
    """
    Simulate trajectory using explicit Euler integration
    """
    wx_rad = rpm_to_rad_s(omega_rpm[0])
    wy_rad = rpm_to_rad_s(omega_rpm[1])
    wz_rad = rpm_to_rad_s(omega_rpm[2])

    vx_init, vy_init, vz_init = initial_velocity_from_angle(v0, alpha, v0z)

    t = [t0]
    x = [x0]
    y = [y0]
    z = [z0]
    vx = [vx_init]
    vy = [vy_init]
    vz = [vz_init]
    speed = [v0]

    i = 0
    while y[i] >= 0.0:
        rho = air_density(y[i], model=density_model)

        if use_drag:
            fdx, fdy, fdz, c_drag = drag_force(vx[i], vy[i], vz[i], speed[i], rho)
            fwx, fwy, fwz = wind_force(c_drag, rho, speed[i], wind)
        else:
            fdx = fdy = fdz = 0.0
            fwx = fwy = fwz = 0.0

        ax_m, ay_m, az_m = magnus_acceleration(vx[i], vy[i], vz[i], wx_rad, wy_rad, wz_rad)

        wind_x, wind_y, wind_z = wind

        # Relative speed is used when aerodynamic effects are active
        if use_drag:
            new_speed = sqrt((vx[i] - wind_x)**2 +
                (vy[i] - wind_y)**2 +
                (vz[i] - wind_z)**2)
        else:
            new_speed = sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)

        # Explicit Euler update
        vx_next = vx[i] - ((fdx / m) - (fwx / m) - ax_m) * dt_local
        vy_next = vy[i] - g * dt_local - ((fdy / m) - (fwy / m) - ay_m) * dt_local
        vz_next = vz[i] - ((fdz / m) - (fwz / m) - az_m) * dt_local

        x_next = x[i] + vx[i] * dt_local
        y_next = y[i] + vy[i] * dt_local
        z_next = z[i] + vz[i] * dt_local
        t_next = t[i] + dt_local

        vx.append(vx_next)
        vy.append(vy_next)
        vz.append(vz_next)
        x.append(x_next)
        y.append(y_next)
        z.append(z_next)
        t.append(t_next)
        speed.append(new_speed)

        i += 1

    return {
        "t": np.array(t),
        "x": np.array(x),
        "y": np.array(y),
        "z": np.array(z),
        "vx": np.array(vx),
        "vy": np.array(vy),
        "vz": np.array(vz),
        "speed": np.array(speed),
    }



def run_for_multiple_angles(
    angles,
    v0,
    v0z,
    omega_rpm,
    use_drag,
    wind,
    density_model
):
    """
    Simulate one trajectory for each launch angle in the input list
    """
    trajectories = []
    labels = []

    for angle in angles:
        data = simulate_trajectory(
            alpha=angle,
            v0=v0,
            v0z=v0z,
            omega_rpm=omega_rpm,
            use_drag=use_drag,
            wind=wind,
            density_model=density_model
        )
        trajectories.append(data)
        labels.append(f"{angle} deg")

    return trajectories, labels



# =============================================================================
# PLOTTING
# =============================================================================

def setup_3d_axis(fig_num=None, title=""):
    """
    Create a 3D axis 
    """
    fig = plt.figure(fig_num)
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("y (m)")
    plt.title(title)
    return fig, ax


def plot_trajectory_set(ax, trajectories, labels, legend_loc=1):
    """
    Plot multiple trajectories on the same 3D axis
    """
    for data, label in zip(trajectories, labels):
        ax.plot3D(data["x"], data["z"], data["y"], label=label)
    ax.legend(loc=legend_loc)



# =============================================================================
# PREDEFINED CASES
# =============================================================================

def case_1():
    """
    Only wz is non-zero, with drag, no wind, constant density
    """
    print("\nCASE 1: only z-component of spin, with drag, no wind.")
    print("Parameters: v0 = 70 m/s, v0z = 0, wz = 1 rpm, wx = wy = 0")

    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(0.0, 0.0, 1.0),
        use_drag=True,
        wind=(0.0, 0.0, 0.0),
        density_model="constant"
    )

    _, ax = setup_3d_axis(title="Magnus effect with spin only along z (no wind)")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()


def case_2():
    """Hook and slice: only wy is non-zero, with opposite signs."""
    print("\nCASE 2: hook and slice (only wy), with drag, no wind.")
    print("Parameters: v0 = 70 m/s, v0z = 0, wx = wz = 0, wy = ±1 rpm")

    # Single launch angle for both spin signs
    trajectories = []
    labels = []
    for wy in [1.0, -1.0]:
        data = simulate_trajectory(
            alpha=45.0,
            v0=70.0,
            v0z=0.0,
            omega_rpm=(0.0, wy, 0.0),
            use_drag=True,
            wind=(0.0, 0.0, 0.0),
            density_model="constant"
        )
        trajectories.append(data)
        labels.append(f"wy = {wy} rpm")

    _, ax = setup_3d_axis(fig_num=1, title="Hook and slice for alpha = 45 deg")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()

    # Multiple angles for wy = +1
    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(0.0, 1.0, 0.0),
        use_drag=True,
        wind=(0.0, 0.0, 0.0),
        density_model="constant"
    )
    _, ax = setup_3d_axis(fig_num=2, title="Slice trajectories for multiple angles (wy = +1 rpm)")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()

    # Multiple angles for wy = -1
    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(0.0, -1.0, 0.0),
        use_drag=True,
        wind=(0.0, 0.0, 0.0),
        density_model="constant"
    )
    _, ax = setup_3d_axis(fig_num=3, title="Hook trajectories for multiple angles (wy = -1 rpm)")
    plot_trajectory_set(ax, trajectories, labels, legend_loc=2)
    plt.show()


def case_3():
    """Three non-zero spin components, with drag, no wind."""
    print("\nCASE 3: three spin components, with drag, no wind.")
    print("Parameters: v0 = 70 m/s, v0z = 0, omega = (2, -4, 3) rpm")

    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(2.0, -4.0, 3.0),
        use_drag=True,
        wind=(0.0, 0.0, 0.0),
        density_model="constant"
    )

    _, ax = setup_3d_axis(title="Magnus effect in all three directions (no wind)")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()


def case_4():
    """Three spin components, with drag and wind."""
    print("\nCASE 4: three spin components, with drag and wind.")
    print("Parameters: v0 = 70 m/s, v0z = 0, omega = (2, -4, 3) rpm, wind = (10, -5, 6) m/s")

    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(2.0, -4.0, 3.0),
        use_drag=True,
        wind=(10.0, -5.0, 6.0),
        density_model="constant"
    )

    _, ax = setup_3d_axis(title="Magnus effect in all three directions with wind")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()


def case_5():
    """Comparison between adiabatic and isothermal density models."""
    print("\nCASE 5: variable air density models.")
    print("Parameters: v0 = 70 m/s, v0z = 0, omega = (2, -4, 3) rpm")

    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(2.0, -4.0, 3.0),
        use_drag=True,
        wind=(0.0, 0.0, 0.0),
        density_model="adiabatic"
    )
    _, ax = setup_3d_axis(fig_num=1, title="Magnus effect + adiabatic density model")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()

    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=70.0,
        v0z=0.0,
        omega_rpm=(2.0, -4.0, 3.0),
        use_drag=True,
        wind=(0.0, 0.0, 0.0),
        density_model="isothermal"
    )
    _, ax = setup_3d_axis(fig_num=2, title="Magnus effect + isothermal density model")
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()



# =============================================================================
# CUSTOM CASE
# =============================================================================

def custom_case():
    """Run a user-defined simulation."""
    print("\nCustom input mode")

    v0 = float(input("Enter initial speed magnitude [m/s]: "))
    wx = float(input("Enter spin component wx [rpm]: "))
    wy = float(input("Enter spin component wy [rpm]: "))
    wz = float(input("Enter spin component wz [rpm]: "))

    use_drag_input = int(input("Enter 1 to include drag, 0 otherwise: "))
    use_drag = (use_drag_input == 1)

    wind = (0.0, 0.0, 0.0)
    density_model = "constant"

    if use_drag:
        wind_x = float(input("Enter wind x-component [m/s]: "))
        wind_y = float(input("Enter wind y-component [m/s]: "))
        wind_z = float(input("Enter wind z-component [m/s]: "))
        wind = (wind_x, wind_y, wind_z)

        approx = int(input(
            "Enter 1 for isothermal density, 2 for adiabatic density, 3 for constant density: "
        ))

        if approx == 1:
            density_model = "isothermal"
        elif approx == 2:
            density_model = "adiabatic"
        elif approx == 3:
            density_model = "constant"
        else:
            raise ValueError("Invalid density model selection.")

    trajectories, labels = run_for_multiple_angles(
        angles=alpha_deg,
        v0=v0,
        v0z=0.0,
        omega_rpm=(wx, wy, wz),
        use_drag=use_drag,
        wind=wind,
        density_model=density_model
    )

    title = (
        f"Custom simulation | drag={use_drag}, "
        f"density={density_model}, wind={wind}, omega=({wx}, {wy}, {wz}) rpm"
    )

    _, ax = setup_3d_axis(title=title)
    plot_trajectory_set(ax, trajectories, labels)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main(mode="default"):
    cases = {
        1: "Only z-component of spin, no wind",
        2: "Hook and slice (wy), no wind",
        3: "Three spin components, no wind",
        4: "Three spin components with wind",
        5: "Isothermal vs adiabatic density models",
        6: "Custom input mode"
    }

    if mode == "interactive":
        print("Available cases:")
        print("1: Only z-component of spin, no wind")
        print("2: Hook and slice (wy), no wind")
        print("3: Three spin components, no wind")
        print("4: Three spin components with wind")
        print("5: Isothermal vs adiabatic density models")
        print("6: Custom input mode")

        case = int(input("Enter the case number (1, 2, 3, 4, 5 or 6): "))

        if case == 1:
            print("\nRunning case 1: Only z-component of spin, no wind\n")
            case_1()
        elif case == 2:
            print("\nRunning case 2: Hook and slice (wy), no wind\n")
            case_2()
        elif case == 3:
            print("\nRunning case 3: Three spin components, no wind\n")
            case_3()
        elif case == 4:
            print("\nRunning case 4: Three spin components with wind\n")
            case_4()
        elif case == 5:
            print("\nRunning case 5: Isothermal vs adiabatic density models\n")
            case_5()
        elif case == 6:
            print("\nRunning case 6: Custom input mode\n")
            custom_case()
        else:
            raise ValueError("Invalid case selection.")

    else:
        case = 2
        description = cases[case]
        print(f"\nRunning case {case}: {description}\n")
        print("Use interactive mode to explore other cases.\n")
        case_2()


if __name__ == "__main__":
    main()