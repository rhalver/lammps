/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(hybrid/kk,AtomVecHybridKokkos);
AtomStyle(hybrid/kk/device,AtomVecHybridKokkos);
AtomStyle(hybrid/kk/host,AtomVecHybridKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_HYBRID_KOKKOS_H
#define LMP_ATOM_VEC_HYBRID_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_hybrid.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecHybridKokkos : public AtomVecKokkos, public AtomVecHybrid {
 public:
  AtomVecHybridKokkos(class LAMMPS *);

  void grow(int) override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  int pack_comm_self(const int &n, const DAT::tdual_int_1d &list,
                     const int nfirst,
                     const int &pbc_flag, const int pbc[]) override;

  int pack_comm_self_fused(const int &n, const DAT::tdual_int_2d_lr &list,
                           const DAT::tdual_int_1d &sendnum_scan,
                           const DAT::tdual_int_1d &firstrecv,
                           const DAT::tdual_int_1d &pbc_flag,
                           const DAT::tdual_int_2d &pbc,
                           const DAT::tdual_int_1d &g2l) override;

  int pack_comm_kokkos(const int &n, const DAT::tdual_int_1d &list,
                       const DAT::tdual_double_2d_lr &buf,
                       const int &pbc_flag, const int pbc[]) override;

  void unpack_comm_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_double_2d_lr &buf) override;

  int pack_comm_vel_kokkos(const int &n, const DAT::tdual_int_1d &list,
                           const DAT::tdual_double_2d_lr &buf,
                           const int &pbc_flag, const int pbc[]) override;

  void unpack_comm_vel_kokkos(const int &n, const int &nfirst,
                              const DAT::tdual_double_2d_lr &buf) override;

  int pack_reverse_self(const int &n, const DAT::tdual_int_1d &list,
                        const int nfirst) override;

  int pack_reverse_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_double_2d_lr &buf) override;

  void unpack_reverse_kokkos(const int &n, const DAT::tdual_int_1d &list,
                             const DAT::tdual_double_2d_lr &buf) override;

  int pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                         DAT::tdual_double_2d_lr buf,
                         int pbc_flag, int *pbc, ExecutionSpace space) override;

  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_double_2d_lr &buf,
                            ExecutionSpace space) override;

  int pack_border_vel_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                             DAT::tdual_double_2d_lr buf,
                             int pbc_flag, int *pbc, ExecutionSpace space) override;

  void unpack_border_vel_kokkos(const int &n, const int &nfirst,
                                const DAT::tdual_double_2d_lr &buf,
                                ExecutionSpace space) override;

  int pack_exchange_kokkos(const int &nsend,DAT::tdual_double_2d_lr &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist,
                           ExecutionSpace space) override;

  int unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf, int nrecv,
                             int nlocal, int dim, double lo, double hi,
                             ExecutionSpace space,
                             DAT::tdual_int_1d &k_indices) override;

  void sync(ExecutionSpace space, uint64_t mask) override;
  void modified(ExecutionSpace space, uint64_t mask) override;
  void sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag = 0) override;

 private:
  DAT::t_tagint_1d d_tag;
  DAT::t_int_1d d_type, d_mask;
  HAT::t_tagint_1d h_tag;
  HAT::t_int_1d h_type, h_mask;

  DAT::t_imageint_1d d_image;
  HAT::t_imageint_1d h_image;

  DAT::t_kkfloat_1d_3_lr d_x;
  DAT::t_kkfloat_1d_3 d_v;
  DAT::t_kkacc_1d_3 d_f;
  HAT::t_kkfloat_1d_3_lr h_x;
  HAT::t_kkfloat_1d_3 h_v;
  HAT::t_kkacc_1d_3 h_f;

  DAT::t_kkfloat_1d_3 d_omega, d_angmom;
  HAT::t_kkfloat_1d_3 h_omega, h_angmom;

  // FULL

  DAT::t_kkfloat_1d d_q;
  HAT::t_kkfloat_1d h_q;

  DAT::t_tagint_1d d_molecule;
  DAT::t_int_2d d_nspecial;
  DAT::t_tagint_2d d_special;
  DAT::t_int_1d d_num_bond;
  DAT::t_int_2d d_bond_type;
  DAT::t_tagint_2d d_bond_atom;

  HAT::t_tagint_1d h_molecule;
  HAT::t_int_2d h_nspecial;
  HAT::t_tagint_2d h_special;
  HAT::t_int_1d h_num_bond;
  HAT::t_int_2d h_bond_type;
  HAT::t_tagint_2d h_bond_atom;

  DAT::t_int_1d d_num_angle;
  DAT::t_int_2d d_angle_type;
  DAT::t_tagint_2d d_angle_atom1,d_angle_atom2,d_angle_atom3;

  HAT::t_int_1d h_num_angle;
  HAT::t_int_2d h_angle_type;
  HAT::t_tagint_2d h_angle_atom1,h_angle_atom2,h_angle_atom3;

  DAT::t_int_1d d_num_dihedral;
  DAT::t_int_2d d_dihedral_type;
  DAT::t_tagint_2d d_dihedral_atom1,d_dihedral_atom2,
    d_dihedral_atom3,d_dihedral_atom4;
  DAT::t_int_1d d_num_improper;
  DAT::t_int_2d d_improper_type;
  DAT::t_tagint_2d d_improper_atom1,d_improper_atom2,
    d_improper_atom3,d_improper_atom4;

  HAT::t_int_1d h_num_dihedral;
  HAT::t_int_2d h_dihedral_type;
  HAT::t_tagint_2d h_dihedral_atom1,h_dihedral_atom2,
    h_dihedral_atom3,h_dihedral_atom4;

  DAT::t_kkfloat_1d_4 d_mu;
  HAT::t_kkfloat_1d_4 h_mu;

  DAT::t_kkfloat_1d_4 d_sp;
  DAT::t_kkacc_1d_3 d_fm;
  DAT::t_kkacc_1d_3 d_fm_long;
  HAT::t_kkfloat_1d_4 h_sp;
  HAT::t_kkacc_1d_3 h_fm;
  HAT::t_kkacc_1d_3 h_fm_long;

  DAT::t_kkfloat_1d d_radius;
  HAT::t_kkfloat_1d h_radius;
  DAT::t_kkfloat_1d d_rmass;
  HAT::t_kkfloat_1d h_rmass;
  DAT::t_kkfloat_1d_3 d_torque;
  HAT::t_kkfloat_1d_3 h_torque;

  DAT::t_kkfloat_1d d_uCond, d_uMech, d_uChem, d_uCG, d_uCGnew,d_rho,d_dpdTheta,d_duChem;
  HAT::t_kkfloat_1d h_uCond, h_uMech, h_uChem, h_uCG, h_uCGnew,h_rho,h_dpdTheta,h_duChem;
};

} // namespace LAMMPS_NS

#endif
#endif
