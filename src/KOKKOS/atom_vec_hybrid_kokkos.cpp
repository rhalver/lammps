// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_hybrid_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "kokkos.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecHybridKokkos::AtomVecHybridKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecHybrid(lmp)
{
  no_comm_vel_flag = 1;
  no_border_vel_flag = 1;
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::grow(int n)
{
  for (int k = 0; k < nstyles; k++) styles[k]->grow(n);
  nmax = atomKK->k_x.view_host().extent(0);

  tag = atom->tag;
  type = atom->type;
  mask = atom->mask;
  image = atom->image;
  x = atom->x;
  v = atom->v;
  f = atom->f;
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecHybridKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  for (int k = 0; k < nstyles; k++)
    (dynamic_cast<AtomVecKokkos*>(styles[k]))->sort_kokkos(Sorter);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecHybridKokkos_PackComm {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d_const _list;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  double _pbc[6];

  AtomVecHybridKokkos_PackComm(
      const typename DAT::ttransform_kkfloat_1d_3_lr &x,
      const typename DAT::tdual_double_2d_lr &buf,
      const typename DAT::tdual_int_1d &list,
      const double &xprd, const double &yprd, const double &zprd,
      const double &xy, const double &xz, const double &yz, const int* const pbc):
      _x(x.view<DeviceType>()),_list(list.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/3;
        const size_t elements = 3;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
        _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
        _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_comm_kokkos(const int &n,
                                          const DAT::tdual_int_1d &list,
                                          const DAT::tdual_double_2d_lr &buf,
                                          const int &pbc_flag,
                                          const int* const pbc)
{
  // Check whether to always run forward communication on the host
  // Choose correct forward PackComm kernel

  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,X_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackComm<LMPHostType,1,1> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackComm<LMPHostType,1,0> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackComm<LMPHostType,0,1> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackComm<LMPHostType,0,0> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    atomKK->sync(Device,X_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackComm<LMPDeviceType,1,1> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackComm<LMPDeviceType,1,0> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackComm<LMPDeviceType,0,1> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackComm<LMPDeviceType,0,0> f(atomKK->k_x,buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  }

  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecHybridKokkos_PackCommSelf {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_kkfloat_1d_3_lr _xw;
  int _nfirst;
  typename AT::t_int_1d_const _list;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  double _pbc[6];

  AtomVecHybridKokkos_PackCommSelf(
      const typename DAT::ttransform_kkfloat_1d_3_lr &x,
      const int &nfirst,
      const typename DAT::tdual_int_1d &list,
      const double &xprd, const double &yprd, const double &zprd,
      const double &xy, const double &xz, const double &yz, const int* const pbc):
      _x(x.view<DeviceType>()),_xw(x.view<DeviceType>()),_nfirst(nfirst),_list(list.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {
        _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
        _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
        const int j = _list(i);
      if (PBC_FLAG == 0) {
          _xw(i+_nfirst,0) = _x(j,0);
          _xw(i+_nfirst,1) = _x(j,1);
          _xw(i+_nfirst,2) = _x(j,2);
      } else {
        if (TRICLINIC == 0) {
          _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd;
          _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd;
          _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
        } else {
          _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
          _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
          _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
        }
      }

  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_comm_self(const int &n, const DAT::tdual_int_1d &list,
                                        const int nfirst, const int &pbc_flag, const int* const pbc) {
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,X_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackCommSelf<LMPHostType,1,1> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackCommSelf<LMPHostType,1,0> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackCommSelf<LMPHostType,0,1> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackCommSelf<LMPHostType,0,0> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
    atomKK->modified(HostKK,X_MASK);
  } else {
    atomKK->sync(Device,X_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackCommSelf<LMPDeviceType,1,1> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackCommSelf<LMPDeviceType,1,0> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackCommSelf<LMPDeviceType,0,1> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackCommSelf<LMPDeviceType,0,0> f(atomKK->k_x,nfirst,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
    atomKK->modified(Device,X_MASK);
  }

  return n*3;
}


/* ---------------------------------------------------------------------- */

template<class DeviceType,int TRICLINIC>
struct AtomVecHybridKokkos_PackCommSelfFused {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_kkfloat_1d_3_lr _xw;
  typename AT::t_int_2d_lr_const _list;
  typename AT::t_int_2d_const _pbc;
  typename AT::t_int_1d_const _pbc_flag;
  typename AT::t_int_1d_const _firstrecv;
  typename AT::t_int_1d_const _sendnum_scan;
  typename AT::t_int_1d_const _g2l;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;

  AtomVecHybridKokkos_PackCommSelfFused(
      const typename DAT::ttransform_kkfloat_1d_3_lr &x,
      const typename DAT::tdual_int_2d_lr &list,
      const typename DAT::tdual_int_2d &pbc,
      const typename DAT::tdual_int_1d &pbc_flag,
      const typename DAT::tdual_int_1d &firstrecv,
      const typename DAT::tdual_int_1d &sendnum_scan,
      const typename DAT::tdual_int_1d &g2l,
      const double &xprd, const double &yprd, const double &zprd,
      const double &xy, const double &xz, const double &yz):
      _x(x.view<DeviceType>()),_xw(x.view<DeviceType>()),
      _list(list.view<DeviceType>()),
      _pbc(pbc.view<DeviceType>()),
      _pbc_flag(pbc_flag.view<DeviceType>()),
      _firstrecv(firstrecv.view<DeviceType>()),
      _sendnum_scan(sendnum_scan.view<DeviceType>()),
      _g2l(g2l.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii) const {

    int iswap = 0;
    while (ii >= _sendnum_scan[iswap]) iswap++;
    int i = ii;
    if (iswap > 0)
      i = ii - _sendnum_scan[iswap-1];

    const int _nfirst = _firstrecv[iswap];
    const int nlocal = _firstrecv[0];

    int j = _list(iswap,i);
    if (j >= nlocal)
      j = _g2l(j-nlocal);

    if (_pbc_flag(ii) == 0) {
        _xw(i+_nfirst,0) = _x(j,0);
        _xw(i+_nfirst,1) = _x(j,1);
        _xw(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      } else {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd + _pbc(ii,5)*_xy + _pbc(ii,4)*_xz;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd + _pbc(ii,3)*_yz;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_comm_self_fused(const int &n, const DAT::tdual_int_2d_lr &list, const DAT::tdual_int_1d &sendnum_scan,
                                         const DAT::tdual_int_1d &firstrecv, const DAT::tdual_int_1d &pbc_flag, const DAT::tdual_int_2d &pbc,
                                         const DAT::tdual_int_1d &g2l) {
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,X_MASK);
    if (domain->triclinic) {
      struct AtomVecHybridKokkos_PackCommSelfFused<LMPHostType,1> f(atomKK->k_x,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecHybridKokkos_PackCommSelfFused<LMPHostType,0> f(atomKK->k_x,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(HostKK,X_MASK);
  } else {
    atomKK->sync(Device,X_MASK);
    if (domain->triclinic) {
      struct AtomVecHybridKokkos_PackCommSelfFused<LMPDeviceType,1> f(atomKK->k_x,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecHybridKokkos_PackCommSelfFused<LMPDeviceType,0> f(atomKK->k_x,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(Device,X_MASK);
  }

  return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_UnpackComm {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_double_2d_lr_const _buf;
  int _first;

  AtomVecHybridKokkos_UnpackComm(
      const typename DAT::ttransform_kkfloat_1d_3_lr &x,
      const typename DAT::tdual_double_2d_lr &buf,
      const int& first):_x(x.view<DeviceType>()),
                        _first(first) {
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/3;
        const size_t elements = 3;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
      };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      _x(i+_first,0) = _buf(i,0);
      _x(i+_first,1) = _buf(i,1);
      _x(i+_first,2) = _buf(i,2);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::unpack_comm_kokkos(const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf) {
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,X_MASK);
    struct AtomVecHybridKokkos_UnpackComm<LMPHostType> f(atomKK->k_x,buf,first);
    Kokkos::parallel_for(n,f);
    atomKK->modified(HostKK,X_MASK);
  } else {
    atomKK->sync(Device,X_MASK);
    struct AtomVecHybridKokkos_UnpackComm<LMPDeviceType> f(atomKK->k_x,buf,first);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,X_MASK);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFORM_VREMAP>
struct AtomVecHybridKokkos_PackCommVel {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_int_1d _mask;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d_const _list;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  double _pbc[6];
  double _h_rate[6];
  const int _deform_vremap;

  AtomVecHybridKokkos_PackCommVel(
    const typename DAT::ttransform_kkfloat_1d_3_lr &x,
    const typename DAT::tdual_int_1d &mask,
    const typename DAT::ttransform_kkfloat_1d_3 &v,
    const typename DAT::tdual_double_2d_lr &buf,
    const typename DAT::tdual_int_1d &list,
    const double &xprd, const double &yprd, const double &zprd,
    const double &xy, const double &xz, const double &yz, const int* const pbc,
    const double * const h_rate,
    const int &deform_vremap):
    _x(x.view<DeviceType>()),
    _mask(mask.view<DeviceType>()),
    _v(v.view<DeviceType>()),
    _list(list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),
    _deform_vremap(deform_vremap)
  {
    const size_t elements = 6;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
    _h_rate[0] = h_rate[0]; _h_rate[1] = h_rate[1]; _h_rate[2] = h_rate[2];
    _h_rate[3] = h_rate[3]; _h_rate[4] = h_rate[4]; _h_rate[5] = h_rate[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
      _buf(i,3) = _v(j,0);
      _buf(i,4) = _v(j,1);
      _buf(i,5) = _v(j,2);
    } else {
      if (TRICLINIC == 0) {
      _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
      _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
      _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
           } else {
      _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
      _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
      _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      }

      if (DEFORM_VREMAP == 0) {
        _buf(i,3) = _v(j,0);
        _buf(i,4) = _v(j,1);
        _buf(i,5) = _v(j,2);
      } else {
        if (_mask(i) & _deform_vremap) {
          _buf(i,3) = _v(j,0) + _pbc[0]*_h_rate[0] + _pbc[5]*_h_rate[5] + _pbc[4]*_h_rate[4];
          _buf(i,4) = _v(j,1) + _pbc[1]*_h_rate[1] + _pbc[3]*_h_rate[3];
          _buf(i,5) = _v(j,2) + _pbc[2]*_h_rate[2];
        } else {
          _buf(i,3) = _v(j,0);
          _buf(i,4) = _v(j,1);
          _buf(i,5) = _v(j,2);
        }
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_comm_vel_kokkos(
  const int &n,
  const DAT::tdual_int_1d &list,
  const DAT::tdual_double_2d_lr &buf,
  const int &pbc_flag,
  const int* const pbc)
{
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,X_MASK|V_MASK);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
          struct AtomVecHybridKokkos_PackCommVel<LMPHostType,1,1,1> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecHybridKokkos_PackCommVel<LMPHostType,1,0,1> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (domain->triclinic) {
          struct AtomVecHybridKokkos_PackCommVel<LMPHostType,1,1,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecHybridKokkos_PackCommVel<LMPHostType,1,0,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackCommVel<LMPHostType,0,1,0> f(
          atomKK->k_x,atomKK->k_mask,
          atomKK->k_v,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackCommVel<LMPHostType,0,0,0> f(
          atomKK->k_x,atomKK->k_mask,
          atomKK->k_v,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    atomKK->sync(Device,X_MASK|V_MASK);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
          struct AtomVecHybridKokkos_PackCommVel<LMPDeviceType,1,1,1> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecHybridKokkos_PackCommVel<LMPDeviceType,1,0,1> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (domain->triclinic) {
          struct AtomVecHybridKokkos_PackCommVel<LMPDeviceType,1,1,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecHybridKokkos_PackCommVel<LMPDeviceType,1,0,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_v,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecHybridKokkos_PackCommVel<LMPDeviceType,0,1,0> f(
          atomKK->k_x,atomKK->k_mask,
          atomKK->k_v,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecHybridKokkos_PackCommVel<LMPDeviceType,0,0,0> f(
          atomKK->k_x,atomKK->k_mask,
          atomKK->k_v,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      }
    }
  }

  return n*6;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_UnpackCommVel {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_double_2d_lr_const _buf;
  int _first;

  AtomVecHybridKokkos_UnpackCommVel(
    const typename DAT::ttransform_kkfloat_1d_3_lr &x,
    const typename DAT::ttransform_kkfloat_1d_3 &v,
    const typename DAT::tdual_double_2d_lr &buf,
    const int& first):
    _x(x.view<DeviceType>()),
    _v(v.view<DeviceType>()),
    _first(first)
  {
    const size_t elements = 6;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _v(i+_first,0) = _buf(i,3);
    _v(i+_first,1) = _buf(i,4);
    _v(i+_first,2) = _buf(i,5);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::unpack_comm_vel_kokkos(const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf) {
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,X_MASK|V_MASK);
    struct AtomVecHybridKokkos_UnpackCommVel<LMPHostType> f(atomKK->k_x,atomKK->k_v,buf,first);
    Kokkos::parallel_for(n,f);
    atomKK->modified(HostKK,X_MASK|V_MASK);
  } else {
    atomKK->sync(Device,X_MASK|V_MASK);
    struct AtomVecHybridKokkos_UnpackCommVel<LMPDeviceType> f(atomKK->k_x,atomKK->k_v,buf,first);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,X_MASK|V_MASK);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_PackReverse {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkacc_1d_3_randomread _f;
  typename AT::t_double_2d_lr _buf;
  int _first;

  AtomVecHybridKokkos_PackReverse(
      const typename DAT::ttransform_kkacc_1d_3 &f,
      const typename DAT::tdual_double_2d_lr &buf,
      const int& first):_f(f.view<DeviceType>()),
                        _first(first) {
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/3;
        const size_t elements = 3;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
      };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _buf(i,0) = _f(i+_first,0);
    _buf(i,1) = _f(i+_first,1);
    _buf(i,2) = _f(i+_first,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_reverse_kokkos(const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf) {
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(HostKK,F_MASK);
    struct AtomVecHybridKokkos_PackReverse<LMPHostType> f(atomKK->k_f,buf,first);
    Kokkos::parallel_for(n,f);
  } else {
    atomKK->sync(Device,F_MASK);
    struct AtomVecHybridKokkos_PackReverse<LMPDeviceType> f(atomKK->k_f,buf,first);
    Kokkos::parallel_for(n,f);
  }

  return n*size_reverse;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_UnPackReverseSelf {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkacc_1d_3_randomread _f;
  typename AT::t_kkacc_1d_3 _fw;
  int _nfirst;
  typename AT::t_int_1d_const _list;

  AtomVecHybridKokkos_UnPackReverseSelf(
      const typename DAT::ttransform_kkacc_1d_3 &f,
      const int &nfirst,
      const typename DAT::tdual_int_1d &list):
      _f(f.view<DeviceType>()),_fw(f.view<DeviceType>()),_nfirst(nfirst),_list(list.view<DeviceType>()) {
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    _fw(j,0) += _f(i+_nfirst,0);
    _fw(j,1) += _f(i+_nfirst,1);
    _fw(j,2) += _f(i+_nfirst,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_reverse_self(const int &n, const DAT::tdual_int_1d &list,
                                     const int nfirst) {
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(HostKK,F_MASK);
    struct AtomVecHybridKokkos_UnPackReverseSelf<LMPHostType> f(atomKK->k_f,nfirst,list);
    Kokkos::parallel_for(n,f);
    atomKK->modified(HostKK,F_MASK);
  } else {
    atomKK->sync(Device,F_MASK);
    struct AtomVecHybridKokkos_UnPackReverseSelf<LMPDeviceType> f(atomKK->k_f,nfirst,list);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,F_MASK);
  }

  return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_UnPackReverse {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkacc_1d_3 _f;
  typename AT::t_double_2d_lr_const _buf;
  typename AT::t_int_1d_const _list;

  AtomVecHybridKokkos_UnPackReverse(
      const typename DAT::ttransform_kkacc_1d_3 &f,
      const typename DAT::tdual_double_2d_lr &buf,
      const typename DAT::tdual_int_1d &list):
      _f(f.view<DeviceType>()),_list(list.view<DeviceType>()) {
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/3;
        const size_t elements = 3;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    _f(j,0) += _buf(i,0);
    _f(j,1) += _buf(i,1);
    _f(j,2) += _buf(i,2);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::unpack_reverse_kokkos(const int &n,
                                          const DAT::tdual_int_1d &list,
                                          const DAT::tdual_double_2d_lr &buf)
{
  // Check whether to always run reverse communication on the host
  // Choose correct reverse UnPackReverse kernel

  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(HostKK,F_MASK);
    struct AtomVecHybridKokkos_UnPackReverse<LMPHostType> f(atomKK->k_f,buf,list);
    Kokkos::parallel_for(n,f);
    atomKK->modified(HostKK,F_MASK);
  } else {
    atomKK->sync(Device,F_MASK);
    struct AtomVecHybridKokkos_UnPackReverse<LMPDeviceType> f(atomKK->k_f,buf,list);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,F_MASK);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG>
struct AtomVecHybridKokkos_PackBorder {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_double_2d_lr _buf;
  const typename AT::t_int_1d_const _list;
  const typename AT::t_kkfloat_1d_3_lr_randomread _x;
  const typename AT::t_tagint_1d _tag;
  const typename AT::t_int_1d _type;
  const typename AT::t_int_1d _mask;
  const typename AT::t_kkfloat_1d _q;
  const typename AT::t_tagint_1d _molecule;
  double _dx,_dy,_dz;

  AtomVecHybridKokkos_PackBorder(
      const typename AT::t_double_2d_lr &buf,
      const typename AT::t_int_1d_const &list,
      const typename AT::t_kkfloat_1d_3_lr &x,
      const typename AT::t_tagint_1d &tag,
      const typename AT::t_int_1d &type,
      const typename AT::t_int_1d &mask,
      const typename AT::t_kkfloat_1d &q,
      const typename AT::t_tagint_1d &molecule,
      const double &dx, const double &dy, const double &dz):
      _buf(buf),_list(list),
      _x(x),_tag(tag),_type(type),_mask(mask),_q(q),_molecule(molecule),
      _dx(dx),_dy(dy),_dz(dz) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      const int j = _list(i);
      if (PBC_FLAG == 0) {
          _buf(i,0) = _x(j,0);
          _buf(i,1) = _x(j,1);
          _buf(i,2) = _x(j,2);
          _buf(i,3) = d_ubuf(_tag(j)).d;
          _buf(i,4) = d_ubuf(_type(j)).d;
          _buf(i,5) = d_ubuf(_mask(j)).d;
          _buf(i,6) = _q(j);
          _buf(i,7) = d_ubuf(_molecule(j)).d;
      } else {
          _buf(i,0) = _x(j,0) + _dx;
          _buf(i,1) = _x(j,1) + _dy;
          _buf(i,2) = _x(j,2) + _dz;
          _buf(i,3) = d_ubuf(_tag(j)).d;
          _buf(i,4) = d_ubuf(_type(j)).d;
          _buf(i,5) = d_ubuf(_mask(j)).d;
          _buf(i,6) = _q(j);
          _buf(i,7) = d_ubuf(_molecule(j)).d;
      }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                               DAT::tdual_double_2d_lr buf,
                                               int pbc_flag, int *pbc, ExecutionSpace space)
{
  double dx,dy,dz;

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (space==Host) {
      AtomVecHybridKokkos_PackBorder<LMPHostType,1> f(
        buf.view_host(), k_sendlist.view_host(),
        h_x,h_tag,h_type,h_mask,h_q,h_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecHybridKokkos_PackBorder<LMPDeviceType,1> f(
        buf.view_device(), k_sendlist.view_device(),
        d_x,d_tag,d_type,d_mask,d_q,d_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }

  } else {
    dx = dy = dz = 0;
    if (space==Host) {
      AtomVecHybridKokkos_PackBorder<LMPHostType,0> f(
        buf.view_host(), k_sendlist.view_host(),
        h_x,h_tag,h_type,h_mask,h_q,h_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecHybridKokkos_PackBorder<LMPDeviceType,0> f(
        buf.view_device(), k_sendlist.view_device(),
        d_x,d_tag,d_type,d_mask,d_q,d_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  }
  return n*size_border;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_UnpackBorder {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  const typename AT::t_double_2d_lr_const _buf;
  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_kkfloat_1d _q;
  typename AT::t_tagint_1d _molecule;
  int _first;


  AtomVecHybridKokkos_UnpackBorder(
      const typename AT::t_double_2d_lr_const &buf,
      typename AT::t_kkfloat_1d_3_lr &x,
      typename AT::t_tagint_1d &tag,
      typename AT::t_int_1d &type,
      typename AT::t_int_1d &mask,
      typename AT::t_kkfloat_1d &q,
      typename AT::t_tagint_1d &molecule,
      const int& first):
    _buf(buf),_x(x),_tag(tag),_type(type),_mask(mask),_q(q),_molecule(molecule),
    _first(first) {
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      _x(i+_first,0) = _buf(i,0);
      _x(i+_first,1) = _buf(i,1);
      _x(i+_first,2) = _buf(i,2);
      _tag(i+_first) = (tagint) d_ubuf(_buf(i,3)).i;
      _type(i+_first) = (int) d_ubuf(_buf(i,4)).i;
      _mask(i+_first) = (int) d_ubuf(_buf(i,5)).i;
      _q(i+_first) = _buf(i,6);
      _molecule(i+_first) = (tagint) d_ubuf(_buf(i,7)).i;
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::unpack_border_kokkos(const int &n, const int &first,
                                                  const DAT::tdual_double_2d_lr &buf,
                                                  ExecutionSpace space) {
  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|Q_MASK|MOLECULE_MASK);

  while (first+n >= nmax) grow(0);

  if (space==Host) {
    struct AtomVecHybridKokkos_UnpackBorder<LMPHostType>
      f(buf.view_host(),h_x,h_tag,h_type,h_mask,h_q,h_molecule,first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecHybridKokkos_UnpackBorder<LMPDeviceType>
      f(buf.view_device(),d_x,d_tag,d_type,d_mask,d_q,d_molecule,first);
    Kokkos::parallel_for(n,f);
  }

  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|Q_MASK|MOLECULE_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecHybridKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_kkfloat_1d_3_randomread _v;
  typename AT::t_tagint_1d_randomread _tag;
  typename AT::t_int_1d_randomread _type;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_imageint_1d_randomread _image;
  typename AT::t_kkfloat_1d_randomread _q;
  typename AT::t_tagint_1d_randomread _molecule;
  typename AT::t_int_2d_randomread _nspecial;
  typename AT::t_tagint_2d_randomread _special;
  typename AT::t_int_1d_randomread _num_bond;
  typename AT::t_int_2d_randomread _bond_type;
  typename AT::t_tagint_2d_randomread _bond_atom;
  typename AT::t_int_1d_randomread _num_angle;
  typename AT::t_int_2d_randomread _angle_type;
  typename AT::t_tagint_2d_randomread _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_int_1d_randomread _num_dihedral;
  typename AT::t_int_2d_randomread _dihedral_type;
  typename AT::t_tagint_2d_randomread _dihedral_atom1,_dihedral_atom2,
    _dihedral_atom3,_dihedral_atom4;
  typename AT::t_int_1d_randomread _num_improper;
  typename AT::t_int_2d_randomread _improper_type;
  typename AT::t_tagint_2d_randomread _improper_atom1,_improper_atom2,
    _improper_atom3,_improper_atom4;
  typename AT::t_kkfloat_1d_3_lr _xw;
  typename AT::t_kkfloat_1d_3 _vw;
  typename AT::t_tagint_1d _tagw;
  typename AT::t_int_1d _typew;
  typename AT::t_int_1d _maskw;
  typename AT::t_imageint_1d _imagew;
  typename AT::t_kkfloat_1d _qw;
  typename AT::t_tagint_1d _moleculew;
  typename AT::t_int_2d _nspecialw;
  typename AT::t_tagint_2d _specialw;
  typename AT::t_int_1d _num_bondw;
  typename AT::t_int_2d _bond_typew;
  typename AT::t_tagint_2d _bond_atomw;
  typename AT::t_int_1d _num_anglew;
  typename AT::t_int_2d _angle_typew;
  typename AT::t_tagint_2d _angle_atom1w,_angle_atom2w,_angle_atom3w;
  typename AT::t_int_1d _num_dihedralw;
  typename AT::t_int_2d _dihedral_typew;
  typename AT::t_tagint_2d _dihedral_atom1w,_dihedral_atom2w,
    _dihedral_atom3w,_dihedral_atom4w;
  typename AT::t_int_1d _num_improperw;
  typename AT::t_int_2d _improper_typew;
  typename AT::t_tagint_2d _improper_atom1w,_improper_atom2w,
    _improper_atom3w,_improper_atom4w;

  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _size_exchange;
  unsigned int _datamask;

  AtomVecHybridKokkos_PackExchangeFunctor(
      const AtomKokkos* atom,
      const DAT::tdual_double_2d_lr buf,
      DAT::tdual_int_1d sendlist,
      DAT::tdual_int_1d copylist,
      const unsigned int datamask):
    _x(atom->k_x.view<DeviceType>()),
    _v(atom->k_v.view<DeviceType>()),
    _tag(atom->k_tag.view<DeviceType>()),
    _type(atom->k_type.view<DeviceType>()),
    _mask(atom->k_mask.view<DeviceType>()),
    _image(atom->k_image.view<DeviceType>()),
    _q(atom->k_q.view<DeviceType>()),
    _molecule(atom->k_molecule.view<DeviceType>()),
    _nspecial(atom->k_nspecial.view<DeviceType>()),
    _special(atom->k_special.view<DeviceType>()),
    _num_bond(atom->k_num_bond.view<DeviceType>()),
    _bond_type(atom->k_bond_type.view<DeviceType>()),
    _bond_atom(atom->k_bond_atom.view<DeviceType>()),
    _num_angle(atom->k_num_angle.view<DeviceType>()),
    _angle_type(atom->k_angle_type.view<DeviceType>()),
    _angle_atom1(atom->k_angle_atom1.view<DeviceType>()),
    _angle_atom2(atom->k_angle_atom2.view<DeviceType>()),
    _angle_atom3(atom->k_angle_atom3.view<DeviceType>()),
    _num_dihedral(atom->k_num_dihedral.view<DeviceType>()),
    _dihedral_type(atom->k_dihedral_type.view<DeviceType>()),
    _dihedral_atom1(atom->k_dihedral_atom1.view<DeviceType>()),
    _dihedral_atom2(atom->k_dihedral_atom2.view<DeviceType>()),
    _dihedral_atom3(atom->k_dihedral_atom3.view<DeviceType>()),
    _dihedral_atom4(atom->k_dihedral_atom4.view<DeviceType>()),
    _num_improper(atom->k_num_improper.view<DeviceType>()),
    _improper_type(atom->k_improper_type.view<DeviceType>()),
    _improper_atom1(atom->k_improper_atom1.view<DeviceType>()),
    _improper_atom2(atom->k_improper_atom2.view<DeviceType>()),
    _improper_atom3(atom->k_improper_atom3.view<DeviceType>()),
    _improper_atom4(atom->k_improper_atom4.view<DeviceType>()),
    _xw(atom->k_x.view<DeviceType>()),
    _vw(atom->k_v.view<DeviceType>()),
    _tagw(atom->k_tag.view<DeviceType>()),
    _typew(atom->k_type.view<DeviceType>()),
    _maskw(atom->k_mask.view<DeviceType>()),
    _imagew(atom->k_image.view<DeviceType>()),
    _qw(atom->k_q.view<DeviceType>()),
    _moleculew(atom->k_molecule.view<DeviceType>()),
    _nspecialw(atom->k_nspecial.view<DeviceType>()),
    _specialw(atom->k_special.view<DeviceType>()),
    _num_bondw(atom->k_num_bond.view<DeviceType>()),
    _bond_typew(atom->k_bond_type.view<DeviceType>()),
    _bond_atomw(atom->k_bond_atom.view<DeviceType>()),
    _num_anglew(atom->k_num_angle.view<DeviceType>()),
    _angle_typew(atom->k_angle_type.view<DeviceType>()),
    _angle_atom1w(atom->k_angle_atom1.view<DeviceType>()),
    _angle_atom2w(atom->k_angle_atom2.view<DeviceType>()),
    _angle_atom3w(atom->k_angle_atom3.view<DeviceType>()),
    _num_dihedralw(atom->k_num_dihedral.view<DeviceType>()),
    _dihedral_typew(atom->k_dihedral_type.view<DeviceType>()),
    _dihedral_atom1w(atom->k_dihedral_atom1.view<DeviceType>()),
    _dihedral_atom2w(atom->k_dihedral_atom2.view<DeviceType>()),
    _dihedral_atom3w(atom->k_dihedral_atom3.view<DeviceType>()),
    _dihedral_atom4w(atom->k_dihedral_atom4.view<DeviceType>()),
    _num_improperw(atom->k_num_improper.view<DeviceType>()),
    _improper_typew(atom->k_improper_type.view<DeviceType>()),
    _improper_atom1w(atom->k_improper_atom1.view<DeviceType>()),
    _improper_atom2w(atom->k_improper_atom2.view<DeviceType>()),
    _improper_atom3w(atom->k_improper_atom3.view<DeviceType>()),
    _improper_atom4w(atom->k_improper_atom4.view<DeviceType>()),
    _sendlist(sendlist.template view<DeviceType>()),
    _copylist(copylist.template view<DeviceType>()),
    _size_exchange(atom->avecKK->size_exchange),
    _datamask(datamask) {
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                             buf.template view<DeviceType>().extent(1))/_size_exchange;
    buffer_view<DeviceType>(_buf,buf,maxsendlist,_size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &mysend) const {
    const int i = _sendlist(mysend);
    _buf(mysend,0) = _size_exchange;
    int m = 1;

    _buf(mysend,m++) = _x(i,0);
    _buf(mysend,m++) = _x(i,1);
    _buf(mysend,m++) = _x(i,2);
    _buf(mysend,m++) = _v(i,0);
    _buf(mysend,m++) = _v(i,1);
    _buf(mysend,m++) = _v(i,2);
    _buf(mysend,m++) = d_ubuf(_tag(i)).d;
    _buf(mysend,m++) = d_ubuf(_type(i)).d;
    _buf(mysend,m++) = d_ubuf(_mask(i)).d;
    _buf(mysend,m++) = d_ubuf(_image(i)).d;

    if (_datamask & Q_MASK)
      _buf(mysend,m++) = _q(i);

    if (_datamask & MOLECULE_MASK)
      _buf(mysend,m++) = d_ubuf(_molecule(i)).d;

    if (_datamask & BOND_MASK) {
      _buf(mysend,m++) = d_ubuf(_num_bond(i)).d;
      for (int k = 0; k < _num_bond(i); k++) {
        _buf(mysend,m++) = d_ubuf(_bond_type(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_bond_atom(i,k)).d;
      }
    }

    if (_datamask & ANGLE_MASK) {
      _buf(mysend,m++) = d_ubuf(_num_angle(i)).d;
      for (int k = 0; k < _num_angle(i); k++) {
        _buf(mysend,m++) = d_ubuf(_angle_type(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_angle_atom1(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_angle_atom2(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_angle_atom3(i,k)).d;
      }
    }

    if (_datamask & DIHEDRAL_MASK) {
      _buf(mysend,m++) = d_ubuf(_num_dihedral(i)).d;
      for (int k = 0; k < _num_dihedral(i); k++) {
        _buf(mysend,m++) = d_ubuf(_dihedral_type(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_dihedral_atom1(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_dihedral_atom2(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_dihedral_atom3(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_dihedral_atom4(i,k)).d;
      }
    }

    if (_datamask & IMPROPER_MASK) {
      _buf(mysend,m++) = d_ubuf(_num_improper(i)).d;
      for (int k = 0; k < _num_improper(i); k++) {
        _buf(mysend,m++) = d_ubuf(_improper_type(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_improper_atom1(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_improper_atom2(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_improper_atom3(i,k)).d;
        _buf(mysend,m++) = d_ubuf(_improper_atom4(i,k)).d;
      }
    }

    if (_datamask & SPECIAL_MASK) {
      _buf(mysend,m++) = d_ubuf(_nspecial(i,0)).d;
      _buf(mysend,m++) = d_ubuf(_nspecial(i,1)).d;
      _buf(mysend,m++) = d_ubuf(_nspecial(i,2)).d;
      for (int k = 0; k < _nspecial(i,2); k++)
        _buf(mysend,m++) = d_ubuf(_special(i,k)).d;
    }

    const int j = _copylist(mysend);

    if (j > -1) {
      _xw(i,0) = _x(j,0);
      _xw(i,1) = _x(j,1);
      _xw(i,2) = _x(j,2);
      _vw(i,0) = _v(j,0);
      _vw(i,1) = _v(j,1);
      _vw(i,2) = _v(j,2);
      _tagw(i) = _tag(j);
      _typew(i) = _type(j);
      _maskw(i) = _mask(j);
      _imagew(i) = _image(j);

      if (_datamask & Q_MASK)
        _qw(i) = _q(j);

      if (_datamask & MOLECULE_MASK)
        _moleculew(i) = _molecule(j);

      if (_datamask & BOND_MASK) {
        _num_bondw(i) = _num_bond(j);
        for (int k = 0; k < _num_bond(j); k++) {
          _bond_typew(i,k) = _bond_type(j,k);
          _bond_atomw(i,k) = _bond_atom(j,k);
        }
      }

      if (_datamask & ANGLE_MASK) {
        _num_anglew(i) = _num_angle(j);
        for (int k = 0; k < _num_angle(j); k++) {
          _angle_typew(i,k) = _angle_type(j,k);
          _angle_atom1w(i,k) = _angle_atom1(j,k);
          _angle_atom2w(i,k) = _angle_atom2(j,k);
          _angle_atom3w(i,k) = _angle_atom3(j,k);
        }
      }

      if (_datamask & DIHEDRAL_MASK) {
        _num_dihedralw(i) = _num_dihedral(j);
        for (int k = 0; k < _num_dihedral(j); k++) {
          _dihedral_typew(i,k) = _dihedral_type(j,k);
          _dihedral_atom1w(i,k) = _dihedral_atom1(j,k);
          _dihedral_atom2w(i,k) = _dihedral_atom2(j,k);
          _dihedral_atom3w(i,k) = _dihedral_atom3(j,k);
          _dihedral_atom4w(i,k) = _dihedral_atom4(j,k);
        }
      }

      if (_datamask & IMPROPER_MASK) {
        _num_improperw(i) = _num_improper(j);
        for (int k = 0; k < _num_improper(j); k++) {
          _improper_typew(i,k) = _improper_type(j,k);
          _improper_atom1w(i,k) = _improper_atom1(j,k);
          _improper_atom2w(i,k) = _improper_atom2(j,k);
          _improper_atom3w(i,k) = _improper_atom3(j,k);
          _improper_atom4w(i,k) = _improper_atom4(j,k);
        }
      }

      if (_datamask & SPECIAL_MASK) {
        _nspecialw(i,0) = _nspecial(j,0);
        _nspecialw(i,1) = _nspecial(j,1);
        _nspecialw(i,2) = _nspecial(j,2);
        for (int k = 0; k < _nspecial(j,2); k++)
          _specialw(i,k) = _special(j,k);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_exchange_kokkos(const int &nsend,DAT::tdual_double_2d_lr &k_buf,
                                                 DAT::tdual_int_1d k_sendlist,
                                                 DAT::tdual_int_1d k_copylist,
                                                 ExecutionSpace space)
{
  if (nsend > (int) (k_buf.view_host().extent(0)*
              k_buf.view_host().extent(1))/size_exchange) {
    int newsize = nsend*size_exchange/k_buf.view_host().extent(1)+1;
    k_buf.resize(newsize,k_buf.view_host().extent(1));
  }
  if (space == HostKK) {
    AtomVecHybridKokkos_PackExchangeFunctor<LMPHostType>
      f(atomKK,k_buf,k_sendlist,k_copylist,datamask_exchange);
    Kokkos::parallel_for(nsend,f);
    return nsend*size_exchange;
  } else {
    AtomVecHybridKokkos_PackExchangeFunctor<LMPDeviceType>
      f(atomKK,k_buf,k_sendlist,k_copylist,datamask_exchange);
    Kokkos::parallel_for(nsend,f);
    return nsend*size_exchange;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int OUTPUT_INDICES>
struct AtomVecHybridKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_kkfloat_1d _q;
  typename AT::t_tagint_1d _molecule;
  typename AT::t_int_2d _nspecial;
  typename AT::t_tagint_2d _special;
  typename AT::t_int_1d _num_bond;
  typename AT::t_int_2d _bond_type;
  typename AT::t_tagint_2d _bond_atom;
  typename AT::t_int_1d _num_angle;
  typename AT::t_int_2d _angle_type;
  typename AT::t_tagint_2d _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_int_1d _num_dihedral;
  typename AT::t_int_2d _dihedral_type;
  typename AT::t_tagint_2d _dihedral_atom1,_dihedral_atom2,
    _dihedral_atom3,_dihedral_atom4;
  typename AT::t_int_1d _num_improper;
  typename AT::t_int_2d _improper_type;
  typename AT::t_tagint_2d _improper_atom1,_improper_atom2,
    _improper_atom3,_improper_atom4;

  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d _nlocal;
  typename AT::t_int_1d _indices;
  int _dim;
  double _lo,_hi;
  int _size_exchange;
  unsigned int _datamask;

  AtomVecHybridKokkos_UnpackExchangeFunctor(
    const AtomKokkos* atom,
    const DAT::tdual_double_2d_lr buf,
    DAT::tdual_int_1d nlocal,
    DAT::tdual_int_1d indices,
    int dim, double lo, double hi,
    unsigned int datamask):
      _x(atom->k_x.view<DeviceType>()),
      _v(atom->k_v.view<DeviceType>()),
      _tag(atom->k_tag.view<DeviceType>()),
      _type(atom->k_type.view<DeviceType>()),
      _mask(atom->k_mask.view<DeviceType>()),
      _image(atom->k_image.view<DeviceType>()),
      _q(atom->k_q.view<DeviceType>()),
      _molecule(atom->k_molecule.view<DeviceType>()),
      _nspecial(atom->k_nspecial.view<DeviceType>()),
      _special(atom->k_special.view<DeviceType>()),
      _num_bond(atom->k_num_bond.view<DeviceType>()),
      _bond_type(atom->k_bond_type.view<DeviceType>()),
      _bond_atom(atom->k_bond_atom.view<DeviceType>()),
      _num_angle(atom->k_num_angle.view<DeviceType>()),
      _angle_type(atom->k_angle_type.view<DeviceType>()),
      _angle_atom1(atom->k_angle_atom1.view<DeviceType>()),
      _angle_atom2(atom->k_angle_atom2.view<DeviceType>()),
      _angle_atom3(atom->k_angle_atom3.view<DeviceType>()),
      _num_dihedral(atom->k_num_dihedral.view<DeviceType>()),
      _dihedral_type(atom->k_dihedral_type.view<DeviceType>()),
      _dihedral_atom1(atom->k_dihedral_atom1.view<DeviceType>()),
      _dihedral_atom2(atom->k_dihedral_atom2.view<DeviceType>()),
      _dihedral_atom3(atom->k_dihedral_atom3.view<DeviceType>()),
      _dihedral_atom4(atom->k_dihedral_atom4.view<DeviceType>()),
      _num_improper(atom->k_num_improper.view<DeviceType>()),
      _improper_type(atom->k_improper_type.view<DeviceType>()),
      _improper_atom1(atom->k_improper_atom1.view<DeviceType>()),
      _improper_atom2(atom->k_improper_atom2.view<DeviceType>()),
      _improper_atom3(atom->k_improper_atom3.view<DeviceType>()),
      _improper_atom4(atom->k_improper_atom4.view<DeviceType>()),
      _nlocal(nlocal.template view<DeviceType>()),
      _indices(indices.template view<DeviceType>()),
      _dim(dim),_lo(lo),_hi(hi),_size_exchange(atom->avecKK->size_exchange),
      _datamask(datamask) {
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                             buf.template view<DeviceType>().extent(1))/_size_exchange;
    buffer_view<DeviceType>(_buf,buf,maxsendlist,_size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &myrecv) const {
    double x = _buf(myrecv,_dim+1);
    int i = -1;
    if (x >= _lo && x < _hi) {
      i = Kokkos::atomic_fetch_add(&_nlocal(0),1);
      int m = 1;
      _x(i,0) = _buf(myrecv,m++);
      _x(i,1) = _buf(myrecv,m++);
      _x(i,2) = _buf(myrecv,m++);
      _v(i,0) = _buf(myrecv,m++);
      _v(i,1) = _buf(myrecv,m++);
      _v(i,2) = _buf(myrecv,m++);
      _tag(i) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      _type(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _mask(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _image(i) = (imageint) d_ubuf(_buf(myrecv,m++)).i;

      if (_datamask & Q_MASK)
        _q(i) = _buf(myrecv,m++);

      if (_datamask & MOLECULE_MASK)
        _molecule(i) = (tagint) d_ubuf(_buf(myrecv,m++)).i;

      if (_datamask & BOND_MASK) {
        _num_bond(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
        for (int k = 0; k < _num_bond(i); k++) {
          _bond_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
          _bond_atom(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        }
      }

      if (_datamask & ANGLE_MASK) {
        _num_angle(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
        for (int k = 0; k < _num_angle(i); k++) {
          _angle_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
          _angle_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _angle_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _angle_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        }
      }

      if (_datamask & DIHEDRAL_MASK) {
        _num_dihedral(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
        for (int k = 0; k < _num_dihedral(i); k++) {
          _dihedral_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
          _dihedral_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _dihedral_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _dihedral_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _dihedral_atom4(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        }
      }

      if (_datamask & IMPROPER_MASK) {
        _num_improper(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
        for (int k = 0; k < _num_improper(i); k++) {
          _improper_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
          _improper_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _improper_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _improper_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          _improper_atom4(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        }
      }

      if (_datamask & SPECIAL_MASK) {
        _nspecial(i,0) = (int) d_ubuf(_buf(myrecv,m++)).i;
        _nspecial(i,1) = (int) d_ubuf(_buf(myrecv,m++)).i;
        _nspecial(i,2) = (int) d_ubuf(_buf(myrecv,m++)).i;
        for (int k = 0; k < _nspecial(i,2); k++)
          _special(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      }
    }

    if (OUTPUT_INDICES)
      _indices(myrecv) = i;
  }
};

/* ---------------------------------------------------------------------- */
int AtomVecHybridKokkos::unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf, int nrecv, int nlocal,
                                              int dim, double lo, double hi, ExecutionSpace space,
                                              DAT::tdual_int_1d &k_indices)
{
  while (nlocal + nrecv/size_exchange >= nmax) grow(0);

  if (space == HostKK) {
    if (k_indices.view_host().data()) {
      k_count.view_host()(0) = nlocal;
      AtomVecHybridKokkos_UnpackExchangeFunctor<LMPHostType,1>
        f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    } else {
      k_count.view_host()(0) = nlocal;
      AtomVecHybridKokkos_UnpackExchangeFunctor<LMPHostType,0>
        f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    }
  } else {
    if (k_indices.view_host().data()) {
      k_count.view_host()(0) = nlocal;
      k_count.modify_host();
      k_count.sync_device();
      AtomVecHybridKokkos_UnpackExchangeFunctor<LMPDeviceType,1>
        f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
      Kokkos::parallel_for(nrecv/size_exchange,f);
      k_count.modify_device();
      k_count.sync_host();
    } else {
      k_count.view_host()(0) = nlocal;
      k_count.modify_host();
      k_count.sync_device();
      AtomVecHybridKokkos_UnpackExchangeFunctor<LMPDeviceType,0>
        f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
      Kokkos::parallel_for(nrecv/size_exchange,f);
      k_count.modify_device();
      k_count.sync_host();
    }
  }

  return k_count.view_host()(0);
}

// TODO: move dynamic_cast into init

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::sync(ExecutionSpace space, unsigned int h_mask)
{
  for (int k = 0; k < nstyles; k++) (dynamic_cast<AtomVecKokkos*>(styles[k]))->sync(space,h_mask);
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::sync_pinned(ExecutionSpace space, unsigned int h_mask, int async_flag)
{
  for (int k = 0; k < nstyles; k++) (dynamic_cast<AtomVecKokkos*>(styles[k]))->sync_pinned(space,h_mask,async_flag);
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::modified(ExecutionSpace space, unsigned int h_mask)
{
  for (int k = 0; k < nstyles; k++) (dynamic_cast<AtomVecKokkos*>(styles[k]))->modified(space,h_mask);
}
