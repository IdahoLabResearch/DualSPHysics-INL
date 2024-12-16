//HEAD_DSPH
/*The revision belongs to Copyright 2024, Battelle Energy Alliance, LLC All Rights Reserved*/
/*
<DUALSPHYSICS>  Copyright (c) 2019 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics.

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

/// \file JSphCpu.cpp \brief Implements the class \ref JSphCpu.

#include "JSphCpu.h"
#include "JCellDivCpu.h"
#include "JPartFloatBi4.h"
#include "Functions.h"
#include "JDsMotion.h"
#include "JArraysCpu.h"
#include "JDsFixedDt.h"
#include "JWaveGen.h"
#include "JMLPistons.h"     //<vs_mlapiston>
#include "JRelaxZones.h"    //<vs_rzone>
#include "JChronoObjects.h" //<vs_chroono>
#include "JDsDamping.h"
#include "JXml.h"
#include "JDsSaveDt.h"
#include "JDsOutputTime.h"
#include "JDsAccInput.h"
#include "JDsGaugeSystem.h"
#include "JSphBoundCorr.h"  //<vs_innlet>
#include <climits>

using namespace std;


//==============================================================================
/// Prepare variables for interaction functions for non-Newtonian formulation.
/// Prepara variables para interaccion.
//==============================================================================
void JSphCpu::ComputePress_NN(unsigned np,unsigned npb) {
  //-Prepare values of rhop for interaction. | Prepara datos derivados de rhop para interaccion.
  const int n=int(np);
#ifdef OMP_USE
#pragma omp parallel for schedule (static) if(n>OMP_LIMIT_COMPUTELIGHT)
#endif
  for(int p=0; p<n; p++) {
    if ((TVisco == VISCO_Hypoplasticity) || (TVisco == VISCO_Elasticity)){
      // directly calculate pressure from previous stress
      Pressc[p]  = -(SpsTauc[p].xx + SpsTauc[p].yy +  SpsTauc[p].zz)/3;
    }else{
      float rhozero_ph; float cteb_ph; float gamma_ph;
      const typecode cod=Codec[p];
      if(CODE_IsFluid(cod)) {
        unsigned cp=CODE_GetTypeValue(cod);
        rhozero_ph=PhaseArray[cp].rho;
        cteb_ph=PhaseArray[cp].CteB;
        gamma_ph=PhaseArray[cp].Gamma;
      }
      else {
        rhozero_ph=CSP.rhopzero;
        cteb_ph=CSP.cteb;
        gamma_ph=CSP.gamma;
      }
      const float rhop=Velrhopc[p].w,rhop_r0=rhop/rhozero_ph;
      Pressc[p]=cteb_ph*(pow(rhop_r0,gamma_ph)-1.0f);
    }
  }
}
//==============================================================================
//Full tensors
//==============================================================================
/// These functions return values for the tensors and invariants.
//==============================================================================
//==============================================================================
/// Calculates the velocity gradient (full matrix)
//==============================================================================
void JSphCpu::GetVelocityGradients_FDA(float rr2,float drx,float dry,float drz
  ,float dvx,float dvy,float dvz,tmatrix3f &dvelp1,float &div_vel)const
{
  //vel gradients
  dvelp1.a11=dvx*drx/rr2; dvelp1.a12=dvx*dry/rr2; dvelp1.a13=dvx*drz/rr2; //Fan et al., 2010
  dvelp1.a21=dvy*drx/rr2; dvelp1.a22=dvy*dry/rr2; dvelp1.a23=dvy*drz/rr2;
  dvelp1.a31=dvz*drx/rr2; dvelp1.a32=dvz*dry/rr2; dvelp1.a33=dvz*drz/rr2;
  div_vel=(dvelp1.a11+dvelp1.a22+dvelp1.a33)/3.f;
}
//==============================================================================
/// Calculates the Strain Rate Tensor (full matrix)
//==============================================================================
void JSphCpu::GetStrainRateTensor(const tmatrix3f &dvelp1,float div_vel,float &I_D,float &II_D,float &J1_D
  ,float &J2_D,float &div_D_tensor,float &D_tensor_magn,tmatrix3f &D_tensor)const
{
  //Strain tensor and invariant
  D_tensor.a11=dvelp1.a11-div_vel;          D_tensor.a12=0.5f*(dvelp1.a12+dvelp1.a21);      D_tensor.a13=0.5f*(dvelp1.a13+dvelp1.a31);
  D_tensor.a21=0.5f*(dvelp1.a21+dvelp1.a12);      D_tensor.a22=dvelp1.a22-div_vel;          D_tensor.a23=0.5f*(dvelp1.a23+dvelp1.a32);
  D_tensor.a31=0.5f*(dvelp1.a31+dvelp1.a13);      D_tensor.a32=0.5f*(dvelp1.a32+dvelp1.a23);      D_tensor.a33=dvelp1.a33-div_vel;
  div_D_tensor=(D_tensor.a11+D_tensor.a22+D_tensor.a33)/3.f;

  //I_D - the first invariant -
  I_D=D_tensor.a11+D_tensor.a22+D_tensor.a33;
  //II_D - the second invariant - expnaded form witout symetry
  float II_D_1=D_tensor.a11*D_tensor.a22+D_tensor.a22*D_tensor.a33+D_tensor.a11*D_tensor.a33;
  float II_D_2=D_tensor.a12*D_tensor.a21+D_tensor.a23*D_tensor.a32+D_tensor.a13*D_tensor.a31;
  II_D=II_D_1-II_D_2;
  //deformation tensor magnitude
  D_tensor_magn=sqrt((II_D*II_D));

  //Main Strain rate invariants
  J1_D=I_D; J2_D=I_D*I_D-2.f*II_D;
}
//==============================================================================
/// Calculates the effective visocity
//==============================================================================
void JSphCpu::GetEta_Effective(const typecode ppx,float tau_yield,float D_tensor_magn,float visco
  ,float m_NN,float n_NN,float &visco_etap1)const
{
  if(D_tensor_magn<=ALMOSTZERO)D_tensor_magn=ALMOSTZERO;
  float miou_yield=(PhaseCte[ppx].tau_max ? PhaseCte[ppx].tau_max/(2.0f*D_tensor_magn) : (tau_yield)/(2.0f*D_tensor_magn)); //HPB will adjust eta		

                                                                                                                                  //if tau_max exists
  bool bi_region=PhaseCte[ppx].tau_max && D_tensor_magn<=PhaseCte[ppx].tau_max/(2.f*PhaseCte[ppx].Bi_multi*visco);
  if(bi_region) { //multiplier
    miou_yield=PhaseCte[ppx].Bi_multi*visco;
  }
  //Papanastasiou
  float miouPap=miou_yield *(1.f-exp(-m_NN*D_tensor_magn));
  float visco_etap1_term1=(PhaseCte[ppx].tau_max ? miou_yield : (miouPap>m_NN*tau_yield||D_tensor_magn==ALMOSTZERO ? m_NN*tau_yield : miouPap));

  //HB
  float miouHB=visco*pow(D_tensor_magn,(n_NN-1.0f));
  //float visco_etap1_term2 = visco;// (miouPap > m_NN*tau_yield ? visco : miouHB);
  float visco_etap1_term2=(bi_region ? visco : (miouPap>m_NN*tau_yield||D_tensor_magn==ALMOSTZERO ? visco : miouHB));

  visco_etap1=visco_etap1_term1+visco_etap1_term2;

  /*
  //use according to you criteria
  - Herein we limit visco_etap1 at very low shear rates
  */
}
//==============================================================================
/// Calculates the stress Tensor (full matrix)
//==============================================================================
void JSphCpu::GetStressTensor(const tmatrix3f &D_tensor,float visco_etap1,float &I_t,float &II_t,float &J1_t,float &J2_t,float &tau_tensor_magn,tmatrix3f &tau_tensor)const
{
  //Stress tensor and invariant   
  tau_tensor.a11=2.f*visco_etap1*(D_tensor.a11);    tau_tensor.a12=2.f*visco_etap1*D_tensor.a12;      tau_tensor.a13=2.f*visco_etap1*D_tensor.a13;
  tau_tensor.a21=2.f*visco_etap1*D_tensor.a21;      tau_tensor.a22=2.f*visco_etap1*(D_tensor.a22);    tau_tensor.a23=2.f*visco_etap1*D_tensor.a23;
  tau_tensor.a31=2.f*visco_etap1*D_tensor.a31;      tau_tensor.a32=2.f*visco_etap1*D_tensor.a32;      tau_tensor.a33=2.f*visco_etap1*(D_tensor.a33);

  //I_t - the first invariant -
  I_t=tau_tensor.a11+tau_tensor.a22+tau_tensor.a33;
  //II_t - the second invariant - expnaded form witout symetry 
  float II_t_1=tau_tensor.a11*tau_tensor.a22+tau_tensor.a22*tau_tensor.a33+tau_tensor.a11*tau_tensor.a33;
  float II_t_2=tau_tensor.a12*tau_tensor.a21+tau_tensor.a23*tau_tensor.a32+tau_tensor.a13*tau_tensor.a31;
  II_t=-II_t_1+II_t_2;
  //stress tensor magnitude
  tau_tensor_magn=sqrt(II_t);
  if(II_t<0.f) {
    printf("****tau_tensor_magn is negative**** \n");
  }
  //Main Strain rate invariants
  J1_t=I_t; J2_t=I_t*I_t-2.f*II_t;
}

//==============================================================================
/// Calculates the Strain Rate Tensor (full matrix)
//==============================================================================
void JSphCpu::GetStrainRateSpinTensors_tsym(const tmatrix3f &dvelp1,tsymatrix3f &D_tensor, tfloat3 &W_tensor)const
{
  //Strain tensor
  D_tensor.xx=dvelp1.a11;  
  D_tensor.yy=dvelp1.a22;  
  D_tensor.zz=dvelp1.a33;
  D_tensor.xy=0.5f*(dvelp1.a12+dvelp1.a21);
  D_tensor.xz=0.5f*(dvelp1.a13+dvelp1.a31);
  D_tensor.yz=0.5f*(dvelp1.a23+dvelp1.a32);  

  //Full spin tensor
  W_tensor.x = 0.5f*(dvelp1.a12-dvelp1.a21);
  W_tensor.y = 0.5f*(dvelp1.a13-dvelp1.a31);
  W_tensor.z = 0.5f*(dvelp1.a23-dvelp1.a32);
}


//==============================================================================
/// Calculates the stress Tensor from hypoplastic model
//==============================================================================
void JSphCpu::GetStressTensor_tsym_hypo(const tsymatrix3f &D_vector, const tfloat3 &W_vector,
 const tsymatrix3f &SigmaOld_vector, float &voidRatio, const double &dt,tsymatrix3f &tau_vector,
 const float &Hypo_angle, const float &Hypo_hs, const float &Hypo_n, const float &Hypo_ed0,
 const float &Hypo_ec0, const float &Hypo_ei0, const float &Hypo_alpha, const float &Hypo_beta,
 const unsigned p1)const
{
  tmatrix3f D_tensor = {D_vector.xx, D_vector.xy, D_vector.xz, 
                        D_vector.xy, D_vector.yy, D_vector.yz, 
                        D_vector.xz, D_vector.yz, D_vector.zz};
  tmatrix3f W_tensor = {0, W_vector.x, W_vector.y,
                        -W_vector.x, 0, W_vector.z,
                        -W_vector.y, -W_vector.z,0};
  tmatrix3f SigmaOld_tensor= {SigmaOld_vector.xx, SigmaOld_vector.xy, SigmaOld_vector.xz, 
                        SigmaOld_vector.xy, SigmaOld_vector.yy, SigmaOld_vector.yz, 
                        SigmaOld_vector.xz, SigmaOld_vector.yz, SigmaOld_vector.zz};
  
  float TrSigma = SigmaOld_tensor.a11+SigmaOld_tensor.a22+SigmaOld_tensor.a33; //Sigma_kk
  // convert stress vector to full stress tensor
  tmatrix3f Ts_tensor;
  Ts_tensor.a11 = SigmaOld_tensor.a11/TrSigma;
  Ts_tensor.a22 = SigmaOld_tensor.a22/TrSigma;
  Ts_tensor.a33 = SigmaOld_tensor.a33/TrSigma;
  Ts_tensor.a12 = SigmaOld_tensor.a12/TrSigma;
  Ts_tensor.a21 = SigmaOld_tensor.a21/TrSigma;
  Ts_tensor.a13 = SigmaOld_tensor.a13/TrSigma;
  Ts_tensor.a31 = SigmaOld_tensor.a31/TrSigma;
  Ts_tensor.a23 = SigmaOld_tensor.a23/TrSigma;
  Ts_tensor.a32 = SigmaOld_tensor.a32/TrSigma;

  tmatrix3f Ts2_tensor;
  Ts2_tensor = Ts_tensor*Ts_tensor;
  float TrTs2 =Ts2_tensor.a11+Ts2_tensor.a22+Ts2_tensor.a33;

  tmatrix3f Tsv_tensor; //normalized deviatoric stress tensor sigmaHeadStar_ij
    float OneOverThree = 1.0/3.0;
  Tsv_tensor.a11 = Ts_tensor.a11 - OneOverThree;
  Tsv_tensor.a12 = Ts_tensor.a12;
  Tsv_tensor.a13 = Ts_tensor.a13;
  Tsv_tensor.a21 = Ts_tensor.a21;
  Tsv_tensor.a22 = Ts_tensor.a22 - OneOverThree;
  Tsv_tensor.a23 = Ts_tensor.a23;
  Tsv_tensor.a31 = Ts_tensor.a31;
  Tsv_tensor.a32 = Ts_tensor.a32;
  Tsv_tensor.a33 = Ts_tensor.a33 - OneOverThree;

  tmatrix3f Tsv2_tensor = Tsv_tensor*Tsv_tensor;
  tmatrix3f Tsv3_tensor = Tsv2_tensor*Tsv_tensor;

  float TrTsv2 = Tsv2_tensor.a11+Tsv2_tensor.a22+Tsv2_tensor.a33;
  float TrTsv3 = Tsv3_tensor.a11+Tsv3_tensor.a22+Tsv3_tensor.a33;
  float sinphi = sin(Hypo_angle*PI/180);
  float sq2 = sqrt(2.0);
  float sq3 = sqrt(3.0);
  float sq6 = sqrt(6.0);
  float c3t; //Lode angle cos(3*theta)
  if (TrTsv2<=1E-10){
    c3t = 1.0;
  }else{
    c3t = -sq6*TrTsv3/pow(TrTsv2, 1.5);
    if (c3t > 1.0)  c3t =  1;
    if (c3t < -1.0) c3t = -1;
  }
  float c1 = sq3*( 3-sinphi )/( 2*sq2*sinphi );
  float c2 = 3*( 3+sinphi )/( 8*sinphi );
  float a1 = 1/(c1+c2*sqrt(TrTsv2)*(1+c3t));

  float ed = Hypo_ed0*exp(-pow((-TrSigma/Hypo_hs),Hypo_n));
  float ec = Hypo_ec0*exp(-pow((-TrSigma/Hypo_hs),Hypo_n));
  float ei = Hypo_ei0*exp(-pow((-TrSigma/Hypo_hs),Hypo_n));

  if (voidRatio<ed){
    // Log->Printf("void ratio e = %f is less than the minimum void ratio at current stress ed = %f ", voidRatio, ed);
    voidRatio = ed;
  }
  if (voidRatio>ei){
    // Log->Printf("void ratio e = %f is more than the maximum void ratio at current stress ei = %f ", voidRatio, ei);
    voidRatio = ei;
  } 

  float fe = pow(ei/voidRatio,Hypo_beta);
  float hi = 1/pow(c1, 2) + OneOverThree - pow((Hypo_ei0-Hypo_ed0)/(Hypo_ec0-Hypo_ed0),Hypo_alpha)/c1/sq3; 
  float fb = Hypo_hs / Hypo_n / hi * (1 + ei) / ei * pow(-TrSigma / Hypo_hs, 1 - Hypo_n); 
  float term5 = Ts_tensor.a11*D_tensor.a11 + Ts_tensor.a12*D_tensor.a12 + Ts_tensor.a13*D_tensor.a13 + Ts_tensor.a21*D_tensor.a21 + Ts_tensor.a22*D_tensor.a22 + Ts_tensor.a23*D_tensor.a23 + Ts_tensor.a31*D_tensor.a31 + Ts_tensor.a32*D_tensor.a32 + Ts_tensor.a33*D_tensor.a33;
  float term6 = sqrt(D_tensor.a11*D_tensor.a11 + D_tensor.a12*D_tensor.a12 + D_tensor.a13*D_tensor.a13 + D_tensor.a21*D_tensor.a21 + D_tensor.a22*D_tensor.a22 + D_tensor.a23*D_tensor.a23 + D_tensor.a31*D_tensor.a31 + D_tensor.a32*D_tensor.a32 + D_tensor.a33*D_tensor.a33);

  float fs = fb*fe;
  float fd = ( voidRatio-ed )/( ec-ed );
  if (fd>0){
    fd=pow(fd, Hypo_alpha);
  }else{
    fd=0;
  }

  tmatrix3f FLL, FNN, sigma_rate, tau_tensor;
  FLL.a11 = pow(a1,2)*D_tensor.a11 + term5*Ts_tensor.a11;
  FLL.a12 = pow(a1,2)*D_tensor.a12 + term5*Ts_tensor.a12;
  FLL.a13 = pow(a1,2)*D_tensor.a13 + term5*Ts_tensor.a13;
  FLL.a21 = pow(a1,2)*D_tensor.a21 + term5*Ts_tensor.a21;
  FLL.a22 = pow(a1,2)*D_tensor.a22 + term5*Ts_tensor.a22;
  FLL.a23 = pow(a1,2)*D_tensor.a23 + term5*Ts_tensor.a23;
  FLL.a31 = pow(a1,2)*D_tensor.a31 + term5*Ts_tensor.a31;
  FLL.a32 = pow(a1,2)*D_tensor.a32 + term5*Ts_tensor.a32;
  FLL.a33 = pow(a1,2)*D_tensor.a33 + term5*Ts_tensor.a33;

  FNN.a11  = a1*(Ts_tensor.a11+Tsv_tensor.a11);
  FNN.a12  = a1*(Ts_tensor.a12+Tsv_tensor.a12);
  FNN.a13  = a1*(Ts_tensor.a13+Tsv_tensor.a13);
  FNN.a21  = a1*(Ts_tensor.a21+Tsv_tensor.a21);
  FNN.a22  = a1*(Ts_tensor.a22+Tsv_tensor.a22);
  FNN.a23  = a1*(Ts_tensor.a23+Tsv_tensor.a23);
  FNN.a31  = a1*(Ts_tensor.a31+Tsv_tensor.a31);
  FNN.a32  = a1*(Ts_tensor.a32+Tsv_tensor.a32);
  FNN.a33  = a1*(Ts_tensor.a33+Tsv_tensor.a33);

  //Stress tensor and invariant   
  sigma_rate.a11 = fs*(FLL.a11+fd*term6*FNN.a11);
  sigma_rate.a12 = fs*(FLL.a12+fd*term6*FNN.a12);
  sigma_rate.a13 = fs*(FLL.a13+fd*term6*FNN.a13);
  sigma_rate.a21 = fs*(FLL.a21+fd*term6*FNN.a21);
  sigma_rate.a22 = fs*(FLL.a22+fd*term6*FNN.a22);
  sigma_rate.a23 = fs*(FLL.a23+fd*term6*FNN.a23);
  sigma_rate.a31 = fs*(FLL.a31+fd*term6*FNN.a31);
  sigma_rate.a32 = fs*(FLL.a32+fd*term6*FNN.a32);
  sigma_rate.a33 = fs*(FLL.a33+fd*term6*FNN.a33);

  sigma_rate = sigma_rate + W_tensor*SigmaOld_tensor - SigmaOld_tensor*W_tensor;

  tau_tensor.a11 = SigmaOld_tensor.a11 + sigma_rate.a11*dt;
  tau_tensor.a12 = SigmaOld_tensor.a12 + sigma_rate.a12*dt;
  tau_tensor.a13 = SigmaOld_tensor.a13 + sigma_rate.a13*dt;
  tau_tensor.a21 = SigmaOld_tensor.a21 + sigma_rate.a21*dt;
  tau_tensor.a22 = SigmaOld_tensor.a22 + sigma_rate.a22*dt;
  tau_tensor.a23 = SigmaOld_tensor.a23 + sigma_rate.a23*dt;
  tau_tensor.a31 = SigmaOld_tensor.a31 + sigma_rate.a31*dt;
  tau_tensor.a32 = SigmaOld_tensor.a32 + sigma_rate.a32*dt;
  tau_tensor.a33 = SigmaOld_tensor.a33 + sigma_rate.a33*dt;

  tau_vector.xx = tau_tensor.a11;
  tau_vector.xy = 0.5f * (tau_tensor.a12 + tau_tensor.a21);
  tau_vector.xz = 0.5f * (tau_tensor.a13 + tau_tensor.a31);
  tau_vector.yz = 0.5f * (tau_tensor.a23 + tau_tensor.a32);
  tau_vector.yy = tau_tensor.a22;
  tau_vector.zz = tau_tensor.a33;

  if (tau_tensor.a11 > 0){
     tau_vector.xx = -10;
     tau_vector.xy = 0;
     tau_vector.xz = 0;
   } else{
     tau_vector.xx = tau_tensor.a11;
     tau_vector.xy = 0.5f*(tau_tensor.a12+tau_tensor.a21);
     tau_vector.xz = 0.5f*(tau_tensor.a13+tau_tensor.a31);
   }
   if (tau_tensor.a22 > 0){
     tau_vector.yz = 0;
     tau_vector.yy = -10;
   } else{
     tau_vector.yz = 0.5f*(tau_tensor.a23+tau_tensor.a32);
     tau_vector.yy = tau_tensor.a22;
   }
   if (tau_tensor.a33 > 0){
     tau_vector.zz = -10;
   } else{
     tau_vector.zz = tau_tensor.a33;
   }

   voidRatio = voidRatio + (1+voidRatio)*(D_tensor.a11 + D_tensor.a22 + D_tensor.a33)*dt  ;
  
  //  check whether void ratio is outside allowed range 
  TrSigma=tau_tensor.a11+tau_tensor.a22+tau_tensor.a33;
  float edd = Hypo_ed0*exp(-pow((-TrSigma/Hypo_hs),Hypo_n))*1.001;
  float eii = Hypo_ei0*exp(-pow((-TrSigma/Hypo_hs),Hypo_n))*0.999;
  if (voidRatio<edd) voidRatio=edd;
  if (voidRatio>eii) voidRatio=eii;

}

//==============================================================================
/// Calculates the stress Tensor from elastic model
//  lameparm2 as "mv" for linear elasticity, lame 2nd parameter 
//  lameparm1 as"lambda" for linear elasticity, lame 1st parameter
//==============================================================================
void JSphCpu::GetStressTensor_tsym_elastic(const tsymatrix3f &D_vector, const tfloat3 &W_vector,
 const tsymatrix3f &SigmaOld_vector, const double &dt,tsymatrix3f &tau_vector, const float &lameparm2,
 const float &lameparm1, const unsigned p1)const
{
  tmatrix3f W_tensor = {0, W_vector.x, W_vector.y,
                        -W_vector.x, 0, W_vector.z,
                        -W_vector.y, -W_vector.z,0};
  tmatrix3f SigmaOld_tensor= {SigmaOld_vector.xx, SigmaOld_vector.xy, SigmaOld_vector.xz, 
                        SigmaOld_vector.xy, SigmaOld_vector.yy, SigmaOld_vector.yz, 
                        SigmaOld_vector.xz, SigmaOld_vector.yz, SigmaOld_vector.zz};
  // sigma_rate = lameparm1 * Identity ^ TrD_tensor + 2 * lameparm2 * D_tensor
  float TrD_tensor = D_vector.xx + D_vector.yy + D_vector.zz; //D_tensor_kk
  tmatrix3f sigma_rate, tau_tensor;
  sigma_rate.a11 = lameparm1 * TrD_tensor + 2 * lameparm2 * D_vector.xx;
  sigma_rate.a12 = 2 * lameparm2 * D_vector.xy;
  sigma_rate.a13 = 2 * lameparm2 * D_vector.xz;
  sigma_rate.a21 = 2 * lameparm2 * D_vector.xy;
  sigma_rate.a22 = lameparm1 * TrD_tensor + 2 * lameparm2 * D_vector.yy;
  sigma_rate.a23 = 2 * lameparm2 * D_vector.yz;
  sigma_rate.a31 = 2 * lameparm2 * D_vector.xz;
  sigma_rate.a32 = 2 * lameparm2 * D_vector.yz;
  sigma_rate.a33 = lameparm1 * TrD_tensor + 2 * lameparm2 * D_vector.zz;
  
  sigma_rate = sigma_rate + W_tensor*SigmaOld_tensor - SigmaOld_tensor*W_tensor;

  tau_tensor.a11 = SigmaOld_tensor.a11 + sigma_rate.a11*dt;
  tau_tensor.a12 = SigmaOld_tensor.a12 + sigma_rate.a12*dt;
  tau_tensor.a13 = SigmaOld_tensor.a13 + sigma_rate.a13*dt;
  tau_tensor.a21 = SigmaOld_tensor.a21 + sigma_rate.a21*dt;
  tau_tensor.a22 = SigmaOld_tensor.a22 + sigma_rate.a22*dt;
  tau_tensor.a23 = SigmaOld_tensor.a23 + sigma_rate.a23*dt;
  tau_tensor.a31 = SigmaOld_tensor.a31 + sigma_rate.a31*dt;
  tau_tensor.a32 = SigmaOld_tensor.a32 + sigma_rate.a32*dt;
  tau_tensor.a33 = SigmaOld_tensor.a33 + sigma_rate.a33*dt;

  tau_vector.xx = tau_tensor.a11;
  tau_vector.xy = 0.5f * (tau_tensor.a12 + tau_tensor.a21);
  tau_vector.xz = 0.5f * (tau_tensor.a13 + tau_tensor.a31);
  tau_vector.yz = 0.5f * (tau_tensor.a23 + tau_tensor.a32);
  tau_vector.yy = tau_tensor.a22;
  tau_vector.zz = tau_tensor.a33;

  // if (tau_tensor.a11 > 0){
  //   tau_vector.xx = -10;
  //   tau_vector.xy = 0;
  //   tau_vector.xz = 0;
  // } else{
  //   tau_vector.xx = tau_tensor.a11;
  //   tau_vector.xy = 0.5f*(tau_tensor.a12+tau_tensor.a21);
  //   tau_vector.xz = 0.5f*(tau_tensor.a13+tau_tensor.a31);
  // }
  // if (tau_tensor.a22 > 0){
  //   tau_vector.yz = 0;
  //   tau_vector.yy = -10;
  // } else{
  //   tau_vector.yz = 0.5f*(tau_tensor.a23+tau_tensor.a32);
  //   tau_vector.yy = tau_tensor.a22;
  // }
  // if (tau_tensor.a33 > 0){
  //   tau_vector.zz = -10;
  // } else{
  //   tau_vector.zz = tau_tensor.a33;
  // }
}
//==============================================================================
//symetric tensors
//==============================================================================
/// Calculates the velocity gradients symetric
//==============================================================================
void JSphCpu::GetVelocityGradients_SPH_tsym(float massp2,const tfloat4 &velrhop2,float dvx,float dvy,float dvz,float frx,float fry,float frz
  ,tmatrix3f &gradvelp1)const
{
  //vel gradients
  const float volp2=-massp2/velrhop2.w;
  float dv=dvx*volp2; gradvelp1.a11+=dv*frx; gradvelp1.a12+=dv*fry; gradvelp1.a13+=dv*frz;
  dv=dvy*volp2;       gradvelp1.a21+=dv*frx; gradvelp1.a22+=dv*fry; gradvelp1.a23+=dv*frz;
  dv=dvz*volp2;       gradvelp1.a31+=dv*frx; gradvelp1.a32+=dv*fry; gradvelp1.a33+=dv*frz;
}
//==============================================================================
/// Calculates the Strain Rate Tensor (symetric)
//==============================================================================
void JSphCpu::GetStrainRateTensor_tsym(const tmatrix3f &dvelp1,float &I_D,float &II_D,float &J1_D,float &J2_D,float &div_D_tensor,float &D_tensor_magn,tsymatrix3f &D_tensor)const
{
  //Strain tensor and invariant
  const float div_vel=(dvelp1.a11+dvelp1.a22+dvelp1.a33)/3.f;
  D_tensor.xx=dvelp1.a11-div_vel;      D_tensor.xy=0.5f*(dvelp1.a12 + dvelp1.a21);     D_tensor.xz=0.5f*(dvelp1.a13 + dvelp1.a31);
  D_tensor.yy=dvelp1.a22-div_vel;      D_tensor.yz=0.5f*(dvelp1.a23  + dvelp1.a32);
  D_tensor.zz=dvelp1.a33-div_vel;
  //the off-diagonal entries of velocity gradients are i.e. 0.5f*(du/dy+dv/dx) with dvelp1.xy=du/dy+dv/dx
  div_D_tensor=(D_tensor.xx+D_tensor.yy+D_tensor.zz)/3.f;
 
  //I_D - the first invariant -
  I_D=D_tensor.xx+D_tensor.yy+D_tensor.zz;
  //II_D - the second invariant - expnaded form witout symetry 
  float II_D_1=D_tensor.xx*D_tensor.yy+D_tensor.yy*D_tensor.zz+D_tensor.xx*D_tensor.zz;
  float II_D_2=D_tensor.xy*D_tensor.xy+D_tensor.yz*D_tensor.yz+D_tensor.xz*D_tensor.xz;
  II_D=-II_D_1+II_D_2;
  ////deformation tensor magnitude
  D_tensor_magn=sqrt((II_D));
  if(II_D<0.f) {
    printf("****D_tensor_magn is negative**** \n");
  }
  //Main Strain rate invariants
  J1_D=I_D; J2_D=I_D*I_D-2.f*II_D;
}
//==============================================================================
/// Calculates the Stress Tensor (symetric)
//==============================================================================
void JSphCpu::GetStressTensor_sym(const tsymatrix3f &D_tensorp1,float visco_etap1,float &I_t,float &II_t,float &J1_t,float &J2_t,float &tau_tensor_magn,tsymatrix3f &tau_tensorp1)const
{
  //Stress tensor and invariant
  tau_tensorp1.xx=2.f*visco_etap1*(D_tensorp1.xx);  tau_tensorp1.xy=2.f*visco_etap1*D_tensorp1.xy;    tau_tensorp1.xz=2.f*visco_etap1*D_tensorp1.xz;
  tau_tensorp1.yy=2.f*visco_etap1*(D_tensorp1.yy);  tau_tensorp1.yz=2.f*visco_etap1*D_tensorp1.yz;
  tau_tensorp1.zz=2.f*visco_etap1*(D_tensorp1.zz);
  //I_t - the first invariant -
  I_t=tau_tensorp1.xx+tau_tensorp1.yy+tau_tensorp1.zz;
  //II_t - the second invariant - expnaded form witout symetry 
  float II_t_1=tau_tensorp1.xx*tau_tensorp1.yy+tau_tensorp1.yy*tau_tensorp1.zz+tau_tensorp1.xx*tau_tensorp1.zz;
  float II_t_2=tau_tensorp1.xy*tau_tensorp1.xy+tau_tensorp1.yz*tau_tensorp1.yz+tau_tensorp1.xz*tau_tensorp1.xz;
  II_t=-II_t_1+II_t_2;
  //stress tensor magnitude
  tau_tensor_magn=sqrt(II_t);
  if(II_t<0.f) {
    printf("****tau_tensor_magn is negative**** \n");
  }
  //Main Stress rate invariants
  J1_t=I_t; J2_t=I_t*I_t-2.f*II_t;
}

//end_of_file