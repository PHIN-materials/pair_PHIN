/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(phin,PairPHIN)

#else

#ifndef LMP_PAIR_PHIN_H
#define LMP_PAIR_PHIN_H

#include "pair.h"

#include <torch/torch.h>

namespace LAMMPS_NS {

class PairPHIN : public Pair {
 public:
  PairPHIN(class LAMMPS *);
  virtual ~PairPHIN();
  virtual void compute(int, int) override;
  void settings(int, char **) override;
  virtual void coeff(int, char **) override;
  virtual double init_one(int, int) override;
  virtual void init_style() override;
  void allocate();
   //   void post_run();

  double cutoff;
  double *uncertainties;
  double tlimit();
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  void *extract_peratom(const char *, int &) override;
  double value, tratio;
  
 protected:
  int nmax;    // allocated size of per-atom arrays
  int * type_mapper;
  int debug_mode = 0;

};

}

#endif
#endif

