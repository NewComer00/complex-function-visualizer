import argparse
import numpy as np
import numpy.matlib
from sympy import lambdify
import matplotlib.pyplot as plt


# from https://stackoverflow.com/a/31364297/15283141
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == '__main__':
    # parse the user-given args
    parser = argparse.ArgumentParser(
        description='Plot the given complex function f(x).')
    parser.add_argument('func_str',
                        type=str,
                        nargs='?',
                        default='exp(x)',
                        help='the function expression, like "exp(x)"')
    parser.add_argument(
        'xr_start',
        type=float,
        nargs='?',
        default=-2,
        help="the start value's real part of the function domain")
    parser.add_argument(
        'xr_end',
        type=float,
        nargs='?',
        default=2,
        help="the end value's real part of the function domain")
    parser.add_argument(
        'xi_start',
        type=float,
        nargs='?',
        default=-2,
        help="the start value's imaginary part of the function domain")
    parser.add_argument(
        'xi_end',
        type=float,
        nargs='?',
        default=2,
        help="the end value's imaginary part of the function domain")
    args = parser.parse_args()

    # the 4d-base for a complex function
    base_xr = np.array([+1, -1, -1]) / np.sqrt(3)
    base_xi = np.array([-1, +1, -1]) / np.sqrt(3)
    base_yr = np.array([+1, +1, +1]) / np.sqrt(3)
    base_yi = np.array([-1, -1, +1]) / np.sqrt(3)
    Base = np.array([base_xr, base_xi, base_yr, base_yi])

    # definition of the complex function
    #f = lambda x: x**2
    f = lambdify('x', args.func_str, 'numpy')

    # domain of the given function
    xr_space = np.linspace(args.xr_start, args.xr_end, 1001)
    xi_space = np.linspace(args.xi_start, args.xi_end, 1001)
    Xr, Xi = np.meshgrid(xr_space, xi_space)
    # range of the function
    Y = f(Xr + 1j * Xi)
    Yr, Yi = Y.real, Y.imag

    Out = Xr[...,np.newaxis] * base_xr \
            + Xi[...,np.newaxis] * base_xi \
            + Yr[...,np.newaxis] * base_yr \
            + Yi[...,np.newaxis] * base_yi

    # plot the complex function
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Surface of the Complex Function f(x)=%s' % args.func_str,
                 fontweight='bold')
    ax.plot_wireframe(Out[..., 0],
                      Out[..., 1],
                      Out[..., 2],
                      linewidth=0.7,
                      label=args.func_str)

    # high light f(x) where x is purely real
    if np.any(xi_space == 0):
        xi_zero_idx = np.argmax(xi_space == 0)
        ax.plot_wireframe(Out[xi_zero_idx, :, 0, np.newaxis],
                          Out[xi_zero_idx, :, 1, np.newaxis],
                          Out[xi_zero_idx, :, 2, np.newaxis],
                          color='r',
                          linewidth=2,
                          label='purely real x')

    # high light f(x) where x is purely imaginary
    if np.any(xr_space == 0):
        xr_zero_idx = np.argmax(xr_space == 0)
        ax.plot_wireframe(Out[:, xr_zero_idx, 0, np.newaxis],
                          Out[:, xr_zero_idx, 1, np.newaxis],
                          Out[:, xr_zero_idx, 2, np.newaxis],
                          color='b',
                          linewidth=2,
                          label='purely imaginary x')

    # plot the 4 axes in 3d
    axis_length = np.max([
        *np.abs(ax.get_ylim()), *np.abs(ax.get_xlim()), *np.abs(ax.get_zlim())
    ])
    ax.quiver(*np.zeros(Base.T.shape),
              *Base.T,
              color='g',
              arrow_length_ratio=0.1,
              length=axis_length,
              label='axes')
    ax.text(*(base_xr * axis_length), "Xr", color='r', label='_nolegend_')
    ax.text(*(base_xi * axis_length), "Xi", color='r', label='_nolegend_')
    ax.text(*(base_yr * axis_length), "Yr", color='r', label='_nolegend_')
    ax.text(*(base_yi * axis_length), "Yi", color='r', label='_nolegend_')

    # display all legends
    plt.legend()
    # all axes should be of the same length and scaling ratio
    set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])

    plt.show()
