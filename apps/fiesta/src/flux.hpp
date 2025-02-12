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

#ifndef FLUX_HPP
#define FLUX_HPP
struct computeFluxWeno2D {
  FS4D var;
  FS2D p;
  FS3D vel;
  FS2D rho;
  FS2D fluxx;
  FS2D fluxy;
  // Kokkos::View<double*> cd;
  FS1D cd;
  int v;
  double eps = 0.000001;

  computeFluxWeno2D(FS4D var_, FS2D p_, FS2D r_, FS3D u_, FS2D fx_, FS2D fy_,
                    FS1D cd_, int v_)
      : var(var_), p(p_), rho(r_), vel(u_), fluxx(fx_), fluxy(fy_), cd(cd_),
        v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {

    int ns = (int)cd(0);
    double ur, vr, w, b1, b2, b3, w1, w2, w3, p1, p2, p3, f1, f2, f3, f4, f5,
        a1, a2;
    double dx = cd(1);
    double dy = cd(2);

    // calculate cell face velocities (in positive direction) with 4th order
    // interpolation velocity is momentum divided by total density
    ur = (-vel(i + 2, j, 0) + 7.0 * vel(i + 1, j, 0) + 7.0 * vel(i, j, 0) -
          vel(i - 1, j, 0)) /
         12.0;
    vr = (-vel(i, j + 2, 1) + 7.0 * vel(i, j + 1, 1) + 7.0 * vel(i, j, 1) -
          vel(i, j - 1, 1)) /
         12.0;

    // for each direction
    for (int idx = 0; idx < 2; ++idx) {
      // get stencil data.  the flux for the energy equation includes pressure
      // so only add pressure for the energy variable (index 2 for 2d problem)
      if (idx == 0) {
        if (ur < 0.0) {
          f1 = var(i + 3, j, 0, v) + (v == 2) * p(i + 3, j);
          f2 = var(i + 2, j, 0, v) + (v == 2) * p(i + 2, j);
          f3 = var(i + 1, j, 0, v) + (v == 2) * p(i + 1, j);
          f4 = var(i, j, 0, v) + (v == 2) * p(i, j);
          f5 = var(i - 1, j, 0, v) + (v == 2) * p(i - 1, j);
        } else {
          f1 = var(i - 2, j, 0, v) + (v == 2) * p(i - 2, j);
          f2 = var(i - 1, j, 0, v) + (v == 2) * p(i - 1, j);
          f3 = var(i, j, 0, v) + (v == 2) * p(i, j);
          f4 = var(i + 1, j, 0, v) + (v == 2) * p(i + 1, j);
          f5 = var(i + 2, j, 0, v) + (v == 2) * p(i + 2, j);
        }
      }
      if (idx == 1) {
        if (vr < 0.0) {
          f1 = var(i, j + 3, 0, v) + (v == 2) * p(i, j + 3);
          f2 = var(i, j + 2, 0, v) + (v == 2) * p(i, j + 2);
          f3 = var(i, j + 1, 0, v) + (v == 2) * p(i, j + 1);
          f4 = var(i, j, 0, v) + (v == 2) * p(i, j);
          f5 = var(i, j - 1, 0, v) + (v == 2) * p(i, j - 1);
        } else {
          f1 = var(i, j - 2, 0, v) + (v == 2) * p(i, j - 2);
          f2 = var(i, j - 1, 0, v) + (v == 2) * p(i, j - 1);
          f3 = var(i, j, 0, v) + (v == 2) * p(i, j);
          f4 = var(i, j + 1, 0, v) + (v == 2) * p(i, j + 1);
          f5 = var(i, j + 2, 0, v) + (v == 2) * p(i, j + 2);
        }
      }

      // calculate weights and other weno stuff

      // b1 = (13/12)*pow((f1-2.0*f2+f3),2.0) +
      // (0.25)*pow((f1-4.0*f2+3.0*f3),2.0); b2 =
      // (13/12)*pow((f2-2.0*f3+f4),2.0)
      // + (0.25)*pow((f2-f4),2.0); b3 = (13/12)*pow((f3-2.0*f4+f5),2.0) +
      // (0.25)*pow((3.0*f3-4.0*f4+f5),2.0);

      // w1 = (0.1)/pow((eps+b1),2.0);
      // w2 = (0.6)/pow((eps+b2),2.0);
      // w3 = (0.3)/pow((eps+b3),2.0);

      // p1 = ( 1.0/3.0)*f1 + (-7.0/6.0)*f2 + (11.0/6.0)*f3;
      // p2 = (-1.0/6.0)*f2 + ( 5.0/6.0)*f3 + ( 1.0/3.0)*f4;
      // p3 = ( 1.0/3.0)*f3 + ( 5.0/6.0)*f4 + (-1.0/6.0)*f5;

      a1 = f1 - 2.0 * f2 + f3;
      a2 = f1 - 4.0 * f2 + 3.0 * f3;
      b1 = (13.0 / 12.0) * a1 * a1 + (0.25) * a2 * a2;
      a1 = f2 - 2.0 * f3 + f4;
      a2 = f2 - f4;
      b2 = (13.0 / 12.0) * a1 * a1 + (0.25) * a2 * a2;
      a1 = f3 - 2.0 * f4 + f5;
      a2 = 3.0 * f3 - 4.0 * f4 + f5;
      b3 = (13.0 / 12.0) * a1 * a1 + (0.25) * a2 * a2;

      a1 = eps + b1;
      w1 = (0.1) / (a1 * a1);
      a1 = eps + b2;
      w2 = (0.6) / (a1 * a1);
      a1 = eps + b3;
      w3 = (0.3) / (a1 * a1);

      p1 = (1.0 / 3.0) * f1 + (-7.0 / 6.0) * f2 + (11.0 / 6.0) * f3;
      p2 = (-1.0 / 6.0) * f2 + (5.0 / 6.0) * f3 + (1.0 / 3.0) * f4;
      p3 = (1.0 / 3.0) * f3 + (5.0 / 6.0) * f4 + (-1.0 / 6.0) * f5;

      w = (w1 * p1 + w2 * p2 + w3 * p3) / (w1 + w2 + w3);

      // calculate weno flux
      if (idx == 0) {
        fluxx(i, j) = ur * w / dx;
      }
      if (idx == 1) {
        fluxy(i, j) = vr * w / dy;
      }
    }
  }
};

struct computeFluxCentered2D {

  FS4D var;
  FS2D p;
  FS2D rho;
  FS2D fluxx;
  FS2D fluxy;
  Kokkos::View<double *> cd;
  int v;

  computeFluxCentered2D(FS4D var_, FS2D p_, FS2D rho_, FS2D fx_, FS2D fy_,
                        Kokkos::View<double *> cd_, int v_)
      : var(var_), p(p_), rho(rho_), fluxx(fx_), fluxy(fy_), cd(cd_), v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {

    double dx = cd(1);
    double dy = cd(2);
    double ur, vr, x1, y1;
    double px, py;
    ur = 0.0;
    vr = 0.0;
    x1 = 0.0;
    y1 = 0.0;

    ur = (0.0 - var(i + 2, j, 0, 0) / rho(i + 2, j) +
          7.0 * var(i + 1, j, 0, 0) / rho(i + 1, j) +
          7.0 * var(i, j, 0, 0) / rho(i, j) -
          var(i - 1, j, 0, 0) / rho(i - 1, j)) /
         12.0;

    vr = (0.0 - var(i, j + 2, 0, 1) / rho(i, j + 2) +
          7.0 * var(i, j + 1, 0, 1) / rho(i, j + 1) +
          7.0 * var(i, j, 0, 1) / rho(i, j) -
          var(i, j - 1, 0, 1) / rho(i, j - 1)) /
         12.0;

    px = (0.0 - p(i + 2, j) + 7.0 * p(i + 1, j) + 7.0 * p(i, j) - p(i - 1, j)) /
         12.0;

    py = (0.0 - p(i, j + 2) + 7.0 * p(i, j + 1) + 7.0 * p(i, j) - p(i, j - 1)) /
         12.0;

    x1 = (0.0 - var(i + 2, j, 0, v) + 7.0 * var(i + 1, j, 0, v) +
          7.0 * var(i, j, 0, v) - var(i - 1, j, 0, v)) /
         12.0;

    y1 = (0.0 - var(i, j + 2, 0, v) + 7.0 * var(i, j + 1, 0, v) +
          7.0 * var(i, j, 0, v) - var(i, j - 1, 0, v)) /
         12.0;

    if (v == 2) {
      x1 = x1 + px;
      y1 = y1 + py;
    }

    fluxx(i, j) = ur * x1 / dx;
    fluxy(i, j) = vr * y1 / dy;
  }
};

struct computeFluxQuick2D {
  FS4D var;
  FS2D p;
  FS2D rho;
  FS2D fluxx;
  FS2D fluxy;
  Kokkos::View<double *> cd;
  int v;
  double eps = 0.000001;

  computeFluxQuick2D(FS4D var_, FS2D p_, FS2D rho_, FS2D fluxx_, FS2D fluxy_,
                     Kokkos::View<double *> cd_, int v_)
      : var(var_), p(p_), rho(rho_), fluxx(fluxx_), fluxy(fluxy_), cd(cd_),
        v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {

    int ns = (int)cd(0);
    double ur, vr, w, f1, f2, f3;
    double dx = cd(1);
    double dy = cd(2);

    // calculate cell face velocities (in positive direction) with 4th order
    // interpolation velocity is momentum divided by total density
    ur = (-var(i + 2, j, 0, 0) / rho(i + 2, j) +
          7.0 * var(i + 1, j, 0, 0) / rho(i + 1, j) +
          7.0 * var(i, j, 0, 0) / rho(i, j) -
          var(i - 1, j, 0, 0) / rho(i - 1, j)) /
         12.0;

    vr = (-var(i, j + 2, 0, 1) / rho(i, j + 2) +
          7.0 * var(i, j + 1, 0, 1) / rho(i, j + 1) +
          7.0 * var(i, j, 0, 1) / rho(i, j) -
          var(i, j - 1, 0, 1) / rho(i, j - 1)) /
         12.0;

    // for each direction
    for (int idx = 0; idx < 2; ++idx) {
      // get stencil data.  the flux for the energy equation includes pressure
      // so only add pressure for the energy variable (index 2 for 2d problem)
      if (idx == 0) {
        if (ur < 0.0) {
          f1 = var(i + 2, j, 0, v) + (v == 2) * p(i + 2, j);
          f2 = var(i + 1, j, 0, v) + (v == 2) * p(i + 1, j);
          f3 = var(i, j, 0, v) + (v == 2) * p(i, j);
        } else {
          f1 = var(i - 1, j, 0, v) + (v == 2) * p(i - 1, j);
          f2 = var(i, j, 0, v) + (v == 2) * p(i, j);
          f3 = var(i + 1, j, 0, v) + (v == 2) * p(i + 1, j);
        }
      }
      if (idx == 1) {
        if (vr < 0.0) {
          f1 = var(i, j + 2, 0, v) + (v == 2) * p(i, j + 2);
          f2 = var(i, j + 1, 0, v) + (v == 2) * p(i, j + 1);
          f3 = var(i, j, 0, v) + (v == 2) * p(i, j);
        } else {
          f1 = var(i, j - 1, 0, v) + (v == 2) * p(i, j - 1);
          f2 = var(i, j, 0, v) + (v == 2) * p(i, j);
          f3 = var(i, j + 1, 0, v) + (v == 2) * p(i, j + 1);
        }
      }

      // quick weights
      w = 0.75 * f2 + 0.375 * f3 - 0.125 * f1;

      // calculate weno flux
      if (idx == 0) {
        fluxx(i, j) = ur * w / dx;
      }
      if (idx == 1) {
        fluxy(i, j) = vr * w / dy;
      }
    }
  }
};
struct calculateWenoFluxesG {

  FS4D var;
  FS3D p;
  FS3D rho;
  FS4D vel;
  FS3D wenox;
  FS3D wenoy;
  FS3D wenoz;
  Kokkos::View<double *> cd;
  int v;
  double eps = 0.000001;

  calculateWenoFluxesG(FS4D var_, FS3D p_, FS3D rho_, FS4D u_, FS3D wenox_,
                       FS3D wenoy_, FS3D wenoz_, Kokkos::View<double *> cd_,
                       int v_)
      : var(var_), p(p_), rho(rho_), vel(u_), wenox(wenox_), wenoy(wenoy_),
        wenoz(wenoz_), cd(cd_), v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {

    int ns = (int)cd(0);
    double ur, vr, wr, w, b1, b2, b3, w1, w2, w3, p1, p2, p3, f1, f2, f3, f4,
        f5, a1, a2;

    ur = (-vel(i + 2, j, k, 0) + 7.0 * vel(i + 1, j, k, 0) +
          7.0 * vel(i, j, k, 0) - vel(i - 1, j, k, 0)) /
         12.0;
    vr = (-vel(i, j + 2, k, 1) + 7.0 * vel(i, j + 1, k, 1) +
          7.0 * vel(i, j, k, 1) - vel(i, j - 1, k, 1)) /
         12.0;
    wr = (-vel(i, j, k + 2, 2) + 7.0 * vel(i, j, k + 1, 2) +
          7.0 * vel(i, j, k, 2) - vel(i, j, k - 1, 2)) /
         12.0;

    for (int idx = 0; idx < 3; ++idx) {
      if (idx == 0) {
        if (ur < 0.0) {
          f1 = var(i + 3, j, k, v) + (v == 3) * p(i + 3, j, k);
          f2 = var(i + 2, j, k, v) + (v == 3) * p(i + 2, j, k);
          f3 = var(i + 1, j, k, v) + (v == 3) * p(i + 1, j, k);
          f4 = var(i, j, k, v) + (v == 3) * p(i, j, k);
          f5 = var(i - 1, j, k, v) + (v == 3) * p(i - 1, j, k);
        } else {
          f1 = var(i - 2, j, k, v) + (v == 3) * p(i - 2, j, k);
          f2 = var(i - 1, j, k, v) + (v == 3) * p(i - 1, j, k);
          f3 = var(i, j, k, v) + (v == 3) * p(i, j, k);
          f4 = var(i + 1, j, k, v) + (v == 3) * p(i + 1, j, k);
          f5 = var(i + 2, j, k, v) + (v == 3) * p(i + 2, j, k);
        }
      }
      if (idx == 1) {
        if (vr < 0.0) {
          f1 = var(i, j + 3, k, v) + (v == 3) * p(i, j + 3, k);
          f2 = var(i, j + 2, k, v) + (v == 3) * p(i, j + 2, k);
          f3 = var(i, j + 1, k, v) + (v == 3) * p(i, j + 1, k);
          f4 = var(i, j, k, v) + (v == 3) * p(i, j, k);
          f5 = var(i, j - 1, k, v) + (v == 3) * p(i, j - 1, k);
        } else {
          f1 = var(i, j - 2, k, v) + (v == 3) * p(i, j - 2, k);
          f2 = var(i, j - 1, k, v) + (v == 3) * p(i, j - 1, k);
          f3 = var(i, j, k, v) + (v == 3) * p(i, j, k);
          f4 = var(i, j + 1, k, v) + (v == 3) * p(i, j + 1, k);
          f5 = var(i, j + 2, k, v) + (v == 3) * p(i, j + 2, k);
        }
      }
      if (idx == 2) {
        if (wr < 0.0) {
          f1 = var(i, j, k + 3, v) + (v == 3) * p(i, j, k + 3);
          f2 = var(i, j, k + 2, v) + (v == 3) * p(i, j, k + 2);
          f3 = var(i, j, k + 1, v) + (v == 3) * p(i, j, k + 1);
          f4 = var(i, j, k, v) + (v == 3) * p(i, j, k);
          f5 = var(i, j, k - 1, v) + (v == 3) * p(i, j, k - 1);
        } else {
          f1 = var(i, j, k - 2, v) + (v == 3) * p(i, j, k - 2);
          f2 = var(i, j, k - 1, v) + (v == 3) * p(i, j, k - 1);
          f3 = var(i, j, k, v) + (v == 3) * p(i, j, k);
          f4 = var(i, j, k + 1, v) + (v == 3) * p(i, j, k + 1);
          f5 = var(i, j, k + 2, v) + (v == 3) * p(i, j, k + 2);
        }
      }

      // b1 = (13/12)*pow((f1-2.0*f2+f3),2.0) +
      // (0.25)*pow((f1-4.0*f2+3.0*f3),2.0); b2 =
      // (13/12)*pow((f2-2.0*f3+f4),2.0)
      // + (0.25)*pow((f2-f4),2.0); b3 = (13/12)*pow((f3-2.0*f4+f5),2.0) +
      // (0.25)*pow((3.0*f3-4.0*f4+f5),2.0);

      // w1 = (0.1)/pow((eps+b1),2.0);
      // w2 = (0.6)/pow((eps+b2),2.0);
      // w3 = (0.3)/pow((eps+b3),2.0);

      // p1 = ( 1.0/3.0)*f1 + (-7.0/6.0)*f2 + (11.0/6.0)*f3;
      // p2 = (-1.0/6.0)*f2 + ( 5.0/6.0)*f3 + ( 1.0/3.0)*f4;
      // p3 = ( 1.0/3.0)*f3 + ( 5.0/6.0)*f4 + (-1.0/6.0)*f5;

      // w = (w1*p1+w2*p2+w3*p3)/(w1+w2+w3);
      a1 = f1 - 2.0 * f2 + f3;
      a2 = f1 - 4.0 * f2 + 3.0 * f3;
      b1 = (13.0 / 12.0) * a1 * a1 + (0.25) * a2 * a2;
      a1 = f2 - 2.0 * f3 + f4;
      a2 = f2 - f4;
      b2 = (13.0 / 12.0) * a1 * a1 + (0.25) * a2 * a2;
      a1 = f3 - 2.0 * f4 + f5;
      a2 = 3.0 * f3 - 4.0 * f4 + f5;
      b3 = (13.0 / 12.0) * a1 * a1 + (0.25) * a2 * a2;

      a1 = eps + b1;
      w1 = (0.1) / (a1 * a1);
      a1 = eps + b2;
      w2 = (0.6) / (a1 * a1);
      a1 = eps + b3;
      w3 = (0.3) / (a1 * a1);

      p1 = (1.0 / 3.0) * f1 + (-7.0 / 6.0) * f2 + (11.0 / 6.0) * f3;
      p2 = (-1.0 / 6.0) * f2 + (5.0 / 6.0) * f3 + (1.0 / 3.0) * f4;
      p3 = (1.0 / 3.0) * f3 + (5.0 / 6.0) * f4 + (-1.0 / 6.0) * f5;

      w = (w1 * p1 + w2 * p2 + w3 * p3) / (w1 + w2 + w3);

      if (idx == 0) {
        wenox(i, j, k) = ur * w / cd(1);
      }
      if (idx == 1) {
        wenoy(i, j, k) = vr * w / cd(2);
      }
      if (idx == 2) {
        wenoz(i, j, k) = wr * w / cd(3);
      }
    }
  }
};
#endif
