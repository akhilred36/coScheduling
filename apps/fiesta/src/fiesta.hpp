/*
  Copyright 2019-2021 The University of New Mexico

  This file is part of FIESTA.
  
  FIESTA is free software: you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License as published by the Free
  Software Foundation, either version 3 of the License, or (at your option) any
  later version.
  
  FIESTA is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
  A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.
  
  You should have received a copy of the GNU Lesser General Public License
  along with FIESTA.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef FIESTA2_HPP
#define FIESTA2_HPP

#include "input.hpp"
#include "writer.hpp"
#include "rkfunction.hpp"
#include <iostream>
#ifndef NOMPI
#include "hdf.hpp"
#else
#include "vtk.hpp"
#endif

namespace Fiesta{
    struct inputConfig initialize(int, char **);
    void initializeSimulation(struct inputConfig&, rk_func*);
    void reportTimers(struct inputConfig&, rk_func*);
    void checkIO(struct inputConfig&, rk_func*, int, double);
    void finalize();
};

#endif
