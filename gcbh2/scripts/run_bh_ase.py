##########################################################################################
# this is an example script to run the Grand Canonical Basin Hopping
# this self-contained script generates the executables to run basin hopping
# you still need to add the chemical potentials, input.traj, and the bh_options.json
##########################################################################################
# generalizations to the code such as general lammmps input file, etc. to come or whatever

import glob, json, argparse
import numpy as np
from ase.io import read
from gcbh2 import GrandCanonicalBasinHopping
from pygcga2 import randomize, mirror_mutate, remove_H, add_H, rand_clustering

atom_elem_to_num = {"H": 1, "C": 6, "O": 8, "Al": 13, "Pt": 78}
from ase.io import read
import numpy as np


def write_opt_file(atom_order):
    # opt.py file
    with open("opt.py", "w") as f:
        f.write("import re\n")
        f.write("import os\n")
        f.write("import glob\n")
        f.write("\n")
        f.write("from ase.io import *\n")
        f.write("from ase.io.trajectory import TrajectoryWriter\n")
        f.write("import numpy as np\n")
        f.write(
            "from ase.calculators.singlepoint import SinglePointCalculator as SPC\n"
        )
        f.write("from ase.constraints import FixAtoms\n")
        #f.write("from pymatgen.io.lammps.data import LammpsData\n")
        f.write("from mace.calculators import mace_mp\n")
        f.write("from ase.md.velocitydistribution import MaxwellBoltzmannDistribution\n")
        f.write("from ase.md.verlet import VelocityVerlet\n")
        f.write("from ase import units\n")

        f.write("def md_mace_universal(surface, calc, T=500, timestep=1, steps=30):\n")
        f.write('    t = surface.copy()\n')
        f.write('    t.set_calculator(calc)\n')
        f.write('    MaxwellBoltzmannDistribution(t, temperature_K=T, force_temp=True)\n')
        f.write('    dyn = VelocityVerlet(t, timestep * units.fs)\n')
        f.write('    dyn.run(steps)\n')
        f.write('    return t\n')

        
        f.write("def main():\n")
        f.write('    atoms = read("./input.traj")\n')
        f.write('    calc = mace_mp(model="large", device="cuda")\n')
        f.write("    images = md_mace_universal(atoms, calc, T=500, timestep=1, steps=30)\n")
        f.write('    images.write("opt.traj")\n')
        f.write("    e = images.get_potential_energy()\n")
        f.write("    f = images.get_forces()\n")
        f.write("    pos = images.get_positions()\n")
        f.write("    posz = pos[:, 2]\n")
        f.write("    ndx = np.where(posz < 5.5)[0]\n")
        f.write("    c = FixAtoms(ndx)\n")
        f.write("    images.set_constraint(c)\n")
        f.write("    images.set_calculator(SPC(images, energy=e, forces=f))\n")
        f.write('    images.write("optimized.traj")\n')
        f.write("main()\n")



def write_optimize_sh():
    with open("optimize.sh", "w") as f:
        f.write("pwd\n")
        #f.write("cp ../../in.opt .\n")
        f.write("cp ../../opt.py .\n")
        f.write("python opt.py\n")


def run_bh(options):
    filescopied = ["opt.py"]
    name = glob.glob(options["input_traj"])
    print(name)
    slab_clean = read(name[0])

    # this is the main part of the code
    bh_run = GrandCanonicalBasinHopping(
        temperature=options["temperature"],
        t_nve=options["t_nve"],
        atoms=slab_clean,
        bash_script="optimize.sh",
        files_to_copied=filescopied,
        restart=True,
        chemical_potential="chemical_potentials.dat",
    )

    dict_bonds = {
        "C-Pt": 1.9,
        "Pt-Pt": 2.869,
        "C-C": 1.54,
        "C-O": 1.43,
        "Pt-O": 2.0,
        "O-O": 1.2,
        "Al-Al": 2.39,
        "H-H": 0.74,
        "H-Pt": 1.89,
        "H-C": 1.09,
        "Al-O": 1.87,
        "Al-Pt": 2.39,
        "Al-H": 1.66,
        "Al-C": 2.13,
        "O-H": 0.96,
    }
    scalar_low = 0.6
    scalar_high = 3
    bond_range = {
        frozenset(("C", "Pt")): [
            dict_bonds["C-Pt"] * scalar_low,
            dict_bonds["C-Pt"] * scalar_high,
        ],
        frozenset(("Pt", "Pt")): [
            dict_bonds["Pt-Pt"] * scalar_low,
            dict_bonds["Pt-Pt"] * scalar_high,
        ],
        frozenset(("C", "C")): [
            dict_bonds["C-C"] * scalar_low,
            dict_bonds["C-C"] * scalar_high,
        ],
        frozenset(("C", "O")): [
            dict_bonds["C-O"] * scalar_low,
            dict_bonds["C-O"] * scalar_high,
        ],
        frozenset(("Pt", "O")): [
            dict_bonds["Pt-O"] * scalar_low,
            dict_bonds["Pt-O"] * scalar_high,
        ],
        frozenset(("O", "O")): [
            dict_bonds["O-O"] * scalar_low,
            dict_bonds["O-O"] * scalar_high,
        ],
        frozenset(("Al", "Al")): [
            dict_bonds["Al-Al"] * scalar_low,
            dict_bonds["Al-Al"] * scalar_high,
        ],
        frozenset(("H", "H")): [
            dict_bonds["H-H"] * scalar_low,
            dict_bonds["H-H"] * scalar_high,
        ],
        frozenset(("H", "Pt")): [
            dict_bonds["H-Pt"] * scalar_low,
            dict_bonds["H-Pt"] * scalar_high,
        ],
        frozenset(("H", "C")): [
            dict_bonds["H-C"] * scalar_low,
            dict_bonds["H-C"] * scalar_high,
        ],
        frozenset(("Al", "O")): [
            dict_bonds["Al-O"] * scalar_low,
            dict_bonds["Al-O"] * scalar_high,
        ],
        frozenset(("Al", "Pt")): [
            dict_bonds["Al-Pt"] * scalar_low,
            dict_bonds["Al-Pt"] * scalar_high,
        ],
        frozenset(("Al", "H")): [
            dict_bonds["Al-H"] * scalar_low,
            dict_bonds["Al-H"] * scalar_high,
        ],
        frozenset(("Al", "C")): [
            dict_bonds["Al-C"] * scalar_low,
            dict_bonds["Al-C"] * scalar_high,
        ],
        frozenset(("O", "H")): [
            dict_bonds["O-H"] * scalar_low,
            dict_bonds["O-H"] * scalar_high,
        ],
    }

    cell = slab_clean.get_cell()
    a = cell[0, 0]
    b = cell[1, 0]
    c = cell[1, 1]
    tol = 1.5
    boundary = np.array(
        [[-tol, -tol], [a + tol, -tol], [a + b + tol, c + tol], [b - tol, c + tol]]
    )

    bh_run.add_modifier(
        randomize,
        name="randomize",
        dr=1,
        bond_range=bond_range,
        max_trial=50,
        weight=1,
        disp_list=["H", "C", "Pt"],
        disp_variance_dict={"H": 0.6, "C": 0.8, "Pt": 1.2},
    )
    # bh_run.add_modifier(nve_n2p2, name="nve",bond_range=bond_range,  z_fix=6, N=100)
    bh_run.add_modifier(mirror_mutate, name="mirror", weight=2)
    bh_run.add_modifier(remove_H, name="remove_H", weight=0.5)
    bh_run.add_modifier(add_H, bond_range=bond_range, max_trial=50, weight=2)
    n_steps = 4000
    bh_run.run(n_steps)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, default="./bh_options.json")
    args = parser.parse_args()
    with open(args.options) as f:
        options = json.load(f)

    atom_order = options["atom_order"]
    write_opt_file(atom_order=atom_order)
    #write_lammps_input_file(atom_order=atom_order)
    write_optimize_sh()
    run_bh(options)


main()
