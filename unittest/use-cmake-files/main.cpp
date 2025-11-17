
#include <mpi.h>
#include <lammps/lammps.h>
#include <lammps/input.h>
#include <lammps/atom.h>
#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
        MPI_Init(&argc, &argv);
        try {
                auto *lmp = new LAMMPS_NS::LAMMPS(argc, argv, MPI_COMM_WORLD);
                lmp->input->file();
                delete lmp;
        } catch (std::exception &) {
        }
        return 0;
}

