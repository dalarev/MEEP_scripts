# Troubleshooting 3d Gaussian DFT field profiles

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

wvl = 1
pol = mp.Ey

sx = 5
sy = 5
sz = 10

resolution = 10

dpml = wvl

cell_size = mp.Vector3(sx + 2 * dpml,
                       sy + 2 * dpml,
                       sz + 2 * dpml)

# sources - GaussianBeamSource
beam_x0 = mp.Vector3()          # beam focus (relative to source center)
beam_kdir = mp.Vector3(z=1)     # beam propagation direction
beam_w0 = sx/8                  # beam waist radius
beam_E0 = mp.Vector3(y=1)
symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1)]
sources = [
    mp.GaussianBeamSource(
        src=mp.GaussianSource(frequency=1/wvl, fwidth=0.1/wvl, is_integrated=True),
        center=mp.Vector3(z=-cell_size.z/2 + dpml),
        size=mp.Vector3(cell_size.x, cell_size.y),
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
    )]

# Simulation
sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    geometry=[],
    boundary_layers=[mp.Absorber(thickness=dpml)],
    sources=sources,
    symmetries=[],
)

# add DFT monitor: XY Source
vol = mp.Volume(
    center=mp.Vector3(z=-cell_size.z/2 + dpml),
    size=mp.Vector3(sx, sy)
)
dft_obj1 = sim.add_dft_fields([pol], 1 / wvl, 0, 1, where=vol)

# add DFT monitor: XY far from source
vol = mp.Volume(
    center=mp.Vector3(z=cell_size.z/2 - dpml - 0.5),
    size=mp.Vector3(sx, sy)
)
dft_obj2 = sim.add_dft_fields([pol], 1 / wvl, 0, 1, where=vol)

# add DFT monitor: YZ cross-section
vol = mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(0, sy, sz)
)
dft_obj3 = sim.add_dft_fields([pol], 1 / wvl, 0, 1, where=vol)

# show YZ cross-section
sim.plot2D(output_plane=mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(0, cell_size.y, cell_size.z))
)
plt.title('YZ Cross-Section')
plt.show()

# run
sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-5))

# get DFT fields: XY Source
x1,y1,z1,w1 = sim.get_array_metadata(dft_cell=dft_obj1)
field_data1 = np.abs(sim.get_dft_array(dft_obj1, pol, 0)) ** 2

# get DFT fields: XY far from source
x2,y2,z2,w2 = sim.get_array_metadata(dft_cell=dft_obj2)
field_data2 = np.abs(sim.get_dft_array(dft_obj2, pol, 0)) ** 2

# get DFT fields: YZ cross-section
x3,y3,z3,w3 = sim.get_array_metadata(dft_cell=dft_obj3)
field_data3 = np.abs(sim.get_dft_array(dft_obj3, pol, 0)) ** 2

# plot DFT fields
def plot_dft_fields(field_data, x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    im = ax.imshow(field_data,
                   interpolation='spline36', cmap='magma', alpha=0.9,
                   extent=[x[0], x[-1], y[0], y[-1]])
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
plot_dft_fields(field_data1, x1, y1, 'x', 'y', 'DFT Fields at source')
plot_dft_fields(field_data2, x2, y2, 'x', 'y', 'DFT Fields far from source')
plot_dft_fields(field_data3, z3, y3, 'z', 'y', 'DFT Fields ZY cross-section')