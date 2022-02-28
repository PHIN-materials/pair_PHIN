/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <compute_nequip.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
//#include <torch/csrc/jit/runtime/graph_executor.h>
//#include <c10/cuda/CUDACachingAllocator.h>


// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9 so the < check is correct.
// This appears to be fixed in 1.10.
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR < 10)
  #define DO_TORCH_FREEZE_HACK
  // For the hack, need more headers:
  #include <torch/csrc/jit/passes/freeze_module.h>
  #include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
  #include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
  #include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#endif


using namespace LAMMPS_NS;

ComputeNEQUIP::ComputeNEQUIP(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg) {

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "NEQUIP is using device " << device << "\n";

  int ntypes = atom->ntypes;

  memory->create(type_mapper, ntypes+1, "pair:type_mapper");


  // compute 1 all nequip model.pth quantity length type1 type2 ... typeN
  if (narg != (6+ntypes))
    error->all(FLERR, "Incorrect args for compute nequip");

  if (strcmp(arg[1], "all") != 0)
    error->all(FLERR, "compute nequip can only operate on group 'all'");

  quantity = arg[4];
  if (screen) fprintf(screen, "compute nequip will evaluate the quantity %s\n", quantity.c_str());

  vector_flag = 1;
  size_vector = std::atoi(arg[5]);
  if(size_vector<=0)
    error->all(FLERR, "Incorrect vector length!");
  memory->create(vector, size_vector, "ComputeNEQUIP:vector");


  // Parse the definition of each atom type
  char **elements = new char*[ntypes+1];
  for (int i = 1; i <= ntypes; i++){
      elements[i] = new char [strlen(arg[i+5])+1];
      strcpy(elements[i], arg[i+5]);
      if (screen) fprintf(screen, "NequIP Coeff: type %d is element %s\n", i, elements[i]);
  }

  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  std::cout << "Loading model from " << arg[3] << "\n";

  std::unordered_map<std::string, std::string> metadata = {
    {"config", ""},
    {"nequip_version", ""},
    {"r_max", ""},
    {"n_species", ""},
    {"type_names", ""},
    {"_jit_bailout_depth", ""},
    {"allow_tf32", ""}
  };
  model = torch::jit::load(std::string(arg[3]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty()) {
    error->all(FLERR, "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?");
  }

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training")) {
    std::cout << "Freezing TorchScript model...\n";
    #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = freeze_module(
        model, {}
      );
      auto graph = model.get_method("forward").graph();
      OptimizeFrozenGraph(graph, optimize_numerics);
      model = out_mod;
    #else
      // Do it normally
      model = torch::jit::freeze(model);
    #endif
  }

  // Set JIT bailout to avoid long recompilations for many steps
  size_t jit_bailout_depth;
  if (metadata["_jit_bailout_depth"].empty()) {
    // This is the default used in the Python code
    jit_bailout_depth = 2;
  } else {
    jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
  }
  torch::jit::getBailoutDepth() = jit_bailout_depth;

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  // std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  // match the type names in the pair_coeff to the metadata
  // to construct a type mapper from LAMMPS type to NequIP atom_types
  int n_species = std::stod(metadata["n_species"]);
  std::stringstream ss;
  ss << metadata["type_names"];
  for (int i = 0; i < n_species; i++){
      char ele[100];
      ss >> ele;
      for (int itype = 1; itype <= ntypes; itype++)
          if (strcmp(elements[itype], ele) == 0)
              type_mapper[itype] = i;
  }

  if (elements){
      for (int i=1; i<ntypes; i++)
          if (elements[i]) delete [] elements[i];
      delete [] elements;
  }

}

ComputeNEQUIP::~ComputeNEQUIP(){
  if (!copymode) {
    memory->destroy(type_mapper);
    memory->destroy(vector);
  }
}

void ComputeNEQUIP::init(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Compute style NEQUIP requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  // TODO: probably also
  neighbor->requests[irequest]->ghost = 0;

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR,"Compute style NEQUIP requires newton pair off");
}

void ComputeNEQUIP::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}



// Force and energy computation
void ComputeNEQUIP::compute_vector(){
  invoked_peratom = update->ntimestep;

  neighbor->build_one(list);

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair==1)
    error->all(FLERR,"Compute style NEQUIP requires 'newton off'");

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = 2*std::accumulate(numneigh, numneigh+inum, 0);

  torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  torch::Tensor tag2type_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  torch::Tensor cell_tensor = torch::zeros({3,3});

  auto pos = pos_tensor.accessor<float, 2>();
  long edges[2*nedges];
  float edge_cell_shifts[3*nedges];
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float,2>();

  // Inverse mapping from tag to "real" atom index
  std::vector<int> tag2i(inum);

  // Loop over real atoms to store tags, types and positions
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    // Inverse mapping from tag to x/f atom index
    tag2i[itag-1] = i; // tag is probably 1-based
    tag2type[itag-1] = type_mapper[itype];
    pos[itag-1][0] = x[i][0];
    pos[itag-1][1] = x[i][1];
    pos[itag-1][2] = x[i][2];
  }

  // Get cell
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  /*
  std::cout << "cell: " << cell_tensor << "\n";
  std::cout << "tag2i: " << "\n";
  for(int itag = 0; itag < inum; itag++){
    std::cout << tag2i[itag] << " ";
  }
  std::cout << std::endl;
  */

  auto cell_inv = cell_tensor.inverse().transpose(0,1);

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  int edge_counter = 0;
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < cutoff*cutoff){
          periodic_shift[0] = x[j][0] - pos[jtag-1][0];
          periodic_shift[1] = x[j][1] - pos[jtag-1][1];
          periodic_shift[2] = x[j][2] - pos[jtag-1][2];

          torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor);
          auto cell_shift = cell_shift_tensor.accessor<float, 1>();
          float * e_vec = &edge_cell_shifts[edge_counter*3];
          e_vec[0] = std::round(cell_shift[0]);
          e_vec[1] = std::round(cell_shift[1]);
          e_vec[2] = std::round(cell_shift[2]);
          //std::cout << "cell shift: " << cell_shift_tensor << "\n";

          // TODO: double check order
          edges[edge_counter*2] = itag - 1; // tag is probably 1-based
          edges[edge_counter*2+1] = jtag - 1; // tag is probably 1-based

          edge_counter++;
      }
    }
  }

  //std::cout << "tag2type: " << tag2type_tensor << "\n";
  //std::cout << "Edges: " << edges_tensor << "\n";
  //std::cout << "Edge _cell_shifts: " << edge_cell_shifts_tensor << "\n";

  // shorten the list before sending to nequip
  torch::Tensor edges_tensor = torch::zeros({2,edge_counter}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({edge_counter,3});
  auto new_edges = edges_tensor.accessor<long, 2>();
  auto new_edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  for (int i=0; i<edge_counter; i++){

      long *e=&edges[i*2];
      new_edges[0][i] = e[0];
      new_edges[1][i] = e[1];

      float *ev = &edge_cell_shifts[i*3];
      new_edge_cell_shifts[i][0] = ev[0];
      new_edge_cell_shifts[i][1] = ev[1];
      new_edge_cell_shifts[i][2] = ev[2];
  }


  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("edge_cell_shift", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("atom_types", tag2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor quantity_tensor = output.at(quantity).toTensor().cpu();
  //std::cout << quantity << ": " << quantity_tensor << std::endl;
  auto quantity = quantity_tensor.data_ptr<float>();

  for(int i = 0; i < size_vector; i++){
    vector[i] = quantity[i];

  }

  /*
  if(device.is_cuda()){
    //torch::cuda::empty_cache();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  */
}

