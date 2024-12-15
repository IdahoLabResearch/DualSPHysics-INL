//HEAD_DSPH
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
/// Perform stress regularization after dedicated time step
//==============================================================================
template<TpKernel tker> void JSphCpu::StressRegularization_SPH
(unsigned n,unsigned pinit,StDivDataCpu divdata,const unsigned *dcell
  ,tsymatrix3f *tau, const tdouble3 *pos,const tfloat4 *velrhop,const typecode *code,const TpVisco tvisco)const
{
  //-Starts execution using OpenMP.
  const int pfin=int(n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  for(int p1=int(pinit); p1<pfin; p1++) {
    float visc=0,arp1=0;
    //-Load data of particle p1. | Carga datos de particula p1.
    const tdouble3 posp1=pos[p1];
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    // const tfloat4 velrhop1=velrhop[p1];
    const typecode pp1=CODE_GetTypeValue(code[p1]);
    tsymatrix3f tau_sump1={0,0,0,0,0,0};
    float ker_sump1=0;
    //---------------------------------------------------------------------------------------------
    //-Search for fluid particle neighbour in adjacent cells.
    const StNgSearch ngs1=nsearch::Init(dcell[p1],false,divdata);
    for(int z=ngs1.zini; z<ngs1.zfin; z++)for(int y=ngs1.yini; y<ngs1.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs1,divdata);

      bool rsym=false; //<vs_syymmetry>
      for(unsigned p2=pif.x; p2<pif.y; p2++) {
        const float drx=float(posp1.x-pos[p2].x);
        float dry=float(posp1.y-pos[p2].y);
        if(rsym)    dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=KernelSize2 && rr2>=ALMOSTZERO) {
          //-Computes kernel.
          const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
          //-obtian tensity
          tfloat4 velrhop2=velrhop[p2];
          //===== Get mass of particle p2 ===== 
          const typecode pp2=CODE_GetTypeValue(code[p2]);
          float massp2;
          if(tvisco==VISCO_Hypoplasticity)massp2=PhaseHypo[pp2].mass;
          if(tvisco==VISCO_Elasticity)massp2=PhaseElastic[pp2].mass;

          const float vol_ker = fac*massp2/velrhop2.w;
          tau_sump1.xx += tau[p2].xx*vol_ker;
          tau_sump1.yy += tau[p2].yy*vol_ker;
          tau_sump1.zz += tau[p2].zz*vol_ker;
          tau_sump1.xy += tau[p2].xy*vol_ker;
          tau_sump1.xz += tau[p2].xz*vol_ker;
          tau_sump1.yz += tau[p2].yz*vol_ker;
          ker_sump1 += vol_ker;
        }
      }
    }
    //---------------------------------------------------------------------------------------------
    //-Search for boundary particle neighbour in adjacent cells.
    /*
    StNgSearch ngs2=nsearch::Init(dcell[p1],true,divdata);
    for(int z=ngs2.zini; z<ngs2.zfin; z++)for(int y=ngs2.yini; y<ngs2.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs2,divdata);
      bool rsym=false; //<vs_syymmetry>
      for(unsigned p2=pif.x; p2<pif.y; p2++) {
        const float drx=float(posp1.x-pos[p2].x);
        float dry=float(posp1.y-pos[p2].y);
        if(rsym)    dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=KernelSize2 && rr2>=ALMOSTZERO) {
          //-Computes kernel.
          const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
          //-obtian density
          tfloat4 velrhop2=velrhop[p2];
          //-obtain boundary particle mass
          float massp2=MassBound;

          const float vol_ker = fac*massp2/velrhop2.w;
          tau_sump1.xx += tau[p2].xx*vol_ker;
          tau_sump1.yy += tau[p2].yy*vol_ker;
          tau_sump1.zz += tau[p2].zz*vol_ker;
          tau_sump1.xy += tau[p2].xy*vol_ker;
          tau_sump1.xz += tau[p2].xz*vol_ker;
          tau_sump1.yz += tau[p2].yz*vol_ker;
          ker_sump1 += vol_ker;
        }
      }
    }
    */
    if (ker_sump1 != 0){
      tau[p1].xx = tau_sump1.xx/ker_sump1;
      tau[p1].yy = tau_sump1.yy/ker_sump1;
      tau[p1].zz = tau_sump1.zz/ker_sump1;
      tau[p1].xy = tau_sump1.xy/ker_sump1;
      tau[p1].xz = tau_sump1.xz/ker_sump1;
      tau[p1].yz = tau_sump1.yz/ker_sump1;
    }
  }
}

//==============================================================================
/// Perform interaction between particles for NN using the SPH approach. Bound-Fluid/Float
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco> void JSphCpu::InteractionForcesBound_NN_SPH
(unsigned n,unsigned pinit,StDivDataCpu divdata,const unsigned *dcell
  ,tsymatrix3f *tau, const tdouble3 *pos, tfloat4 *velrhop,const typecode *code,const unsigned *idp
  ,float &viscdt,float *ar, double &dt)const
{
  //-Initialize viscth to calculate max viscdt with OpenMP. | Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscth[OMP_MAXTHREADS*OMP_STRIDE];
  for(int th=0; th<OmpThreads; th++)viscth[th*OMP_STRIDE]=0;
  //-Starts execution using OpenMP.
  const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  for(int p1=int(pinit); p1<pfin; p1++) {
    float visc=0,arp1=0;
    //-Load data of particle p1. | Carga datos de particula p1.
    const tdouble3 posp1=pos[p1];
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    const tfloat4 velrhop1=velrhop[p1];
    tsymatrix3f tau_sump1={0,0,0,0,0,0};
    tfloat4 relative_pos_sump1={0,0,0,0};
    tfloat4 velrhop1_sum={0,0,0,0};
    //-Search for neighbours in adjacent cells.
    const StNgSearch ngs=nsearch::Init(dcell[p1],false,divdata);
    for(int z=ngs.zini; z<ngs.zfin; z++)for(int y=ngs.yini; y<ngs.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs,divdata);

      //-Interaction of boundary with type Fluid/Float | Interaccion de Bound con varias Fluid/Float.
      //---------------------------------------------------------------------------------------------
      bool rsym=false; //<vs_syymmetry>
      for(unsigned p2=pif.x; p2<pif.y; p2++) {
        const float drx=float(posp1.x-pos[p2].x);
        float dry=float(posp1.y-pos[p2].y);
        if(rsym)    dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=KernelSize2 && rr2>=ALMOSTZERO) {
          //-Computes kernel.
          const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
          const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

           //===== Get mass of particle p2 ===== 
          const typecode pp2=CODE_GetTypeValue(code[p2]);
          //<vs_non-Newtonian>
          float massp2;
          if(tvisco==VISCO_Hypoplasticity){
            massp2=PhaseHypo[pp2].mass; //-Contiene masa de particula segun sea bound o fluid.
            //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);
          }else if(tvisco==VISCO_Elasticity){
            massp2=PhaseElastic[pp2].mass;
          }else{
            massp2=PhaseArray[pp2].mass; //-Contiene masa de particula segun sea bound o fluid.
          }

          bool compute=true;      //-Deactivate when using DEM and/or bound-float. | Se desactiva cuando se usa DEM y es bound-float.
          if(USE_FLOATING) {
            bool ftp2=CODE_IsFloating(code[p2]);
            if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
            compute=!(USE_FTEXTERNAL && ftp2); //-Deactivate when using DEM/Chrono and/or bound-float. | Se desactiva cuando se usa DEM/Chrono y es bound-float.
          }

          if(compute) {
            //-Density derivative (Continuity equation).
            tfloat4 velrhop2=velrhop[p2];
            if(rsym)velrhop2.y=-velrhop2.y; //<vs_syymmetry>
            const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
            if(compute)arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w);

            if ((tvisco == VISCO_Hypoplasticity) || (tvisco == VISCO_Elasticity)){
              const float vol_ker = fac*massp2/velrhop2.w;
              tau_sump1.xx += tau[p2].xx*vol_ker;
              tau_sump1.yy += tau[p2].yy*vol_ker;
              tau_sump1.zz += tau[p2].zz*vol_ker;
              tau_sump1.xy += tau[p2].xy*vol_ker;
              tau_sump1.xz += tau[p2].xz*vol_ker;
              tau_sump1.yz += tau[p2].yz*vol_ker;
              
              velrhop1_sum.x +=  velrhop2.x*vol_ker;
              velrhop1_sum.y +=  velrhop2.y*vol_ker;
              velrhop1_sum.z +=  velrhop2.z*vol_ker;

              relative_pos_sump1.x += drx*velrhop2.w*vol_ker;
              relative_pos_sump1.y += dry*velrhop2.w*vol_ker;
              relative_pos_sump1.z += drz*velrhop2.w*vol_ker;
              relative_pos_sump1.w += vol_ker;
            }

            {//-Viscosity.
              const float dot=drx*dvx+dry*dvy+drz*dvz;
              const float dot_rr2=dot/(rr2+Eta2);
              visc=max(dot_rr2,visc);
            }
          }
          rsym=(rsymp1&&!rsym && float(posp1.y-dry)<=KernelSize); //<vs_syymmetry>
          if(rsym)p2--;                                             //<vs_syymmetry>
        }
        else rsym=false;                                            //<vs_syymmetry>
      }
    }
    //-Sum results together. | Almacena resultados.
    if(arp1||visc) {
      ar[p1]+=arp1;
      const int th=omp_get_thread_num();
      if(visc>viscth[th*OMP_STRIDE])viscth[th*OMP_STRIDE]=visc;
    }
    if (((tvisco == VISCO_Hypoplasticity) || (tvisco == VISCO_Elasticity)) & (relative_pos_sump1.w != 0) ){
      tau[p1].xx = (tau_sump1.xx - (Gravity.x-velrhop1.x/dt)*relative_pos_sump1.x)/relative_pos_sump1.w;
      tau[p1].yy = (tau_sump1.yy - (Gravity.y-velrhop1.y/dt)*relative_pos_sump1.y)/relative_pos_sump1.w;
      tau[p1].zz = (tau_sump1.zz - (Gravity.z-velrhop1.z/dt)*relative_pos_sump1.z)/relative_pos_sump1.w;
      // tau[p1].xx = (tau_sump1.xx - velrhop1.x/dt*relative_pos_sump1.x)/relative_pos_sump1.w;
      // tau[p1].yy = (tau_sump1.yy - velrhop1.y/dt*relative_pos_sump1.y)/relative_pos_sump1.w;
      // tau[p1].zz = (tau_sump1.zz - velrhop1.z/dt*relative_pos_sump1.z)/relative_pos_sump1.w;
      tau[p1].xy = tau_sump1.xy /relative_pos_sump1.w;
      tau[p1].xz = tau_sump1.xz /relative_pos_sump1.w;
      tau[p1].yz = tau_sump1.yz /relative_pos_sump1.w;
      velrhop[p1].x = 2*velrhop1.x - velrhop1_sum.x/relative_pos_sump1.w;
      velrhop[p1].y = 2*velrhop1.y - velrhop1_sum.y/relative_pos_sump1.w;
      velrhop[p1].z = 2*velrhop1.z - velrhop1_sum.z/relative_pos_sump1.w;
    }

    // if (p1 == 6808 || p1 == 3059) {
    //     Log->Printf("particle %d", p1);
    //     Log->Printf("After running Bound_NN_sph, arp1 = %f", arp1);
    // }
  }
  //-Keep max value in viscdt. | Guarda en viscdt el valor maximo.
  for(int th=0; th<OmpThreads; th++)if(viscdt<viscth[th*OMP_STRIDE])viscdt=viscth[th*OMP_STRIDE];
}

//==============================================================================
/// Perform calculations for NN using the SPH approach: Calculates the Stress tensor
//==============================================================================
template<TpFtMode ftmode,TpVisco tvisco> void JSphCpu::InteractionForcesFluid_NN_SPH_Visco_Stress_tensor
(unsigned n,unsigned pinit,float visco,float *visco_eta, float *void_ratio,
  tsymatrix3f *tau,const tmatrix3f* gradvel,const tsymatrix3f *D_tensor,const tfloat3 *W_tensor,
  const tfloat4 *velrhop, float *auxnn ,const tdouble3 *pos,
  const typecode *code,const unsigned *idp, double &dt)const
{
  //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
  const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  for(int p1=int(pinit); p1<pfin; p1++) {
    //-Obtain data of particle p1.
    const tdouble3 posp1=pos[p1];
    //stress related to p1
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    const typecode pp1=CODE_GetTypeValue(code[p1]);
    tsymatrix3f tau_tensorp1={0,0,0,0,0,0};
    float voidRatio = 0;

    if ((tvisco == VISCO_Hypoplasticity) || (tvisco == VISCO_Elasticity)){
      const tmatrix3f gradvelp1=gradvel[p1];
      // const tfloat3 W_tensorp1=W_tensor[p1];
      const tsymatrix3f SigmaOld_vectorp1=tau[p1];
      //if(p1==11163)   Log->Printf("SigmaOld_vector of 11163 is: %f %f %f %f %f %f",SigmaOld_vectorp1.xx,SigmaOld_vectorp1.xy,SigmaOld_vectorp1.xz,SigmaOld_vectorp1.yy,SigmaOld_vectorp1.yz,SigmaOld_vectorp1.zz);
      // const tfloat4 velrhop1=velrhop[p1];
      if(tvisco == VISCO_Hypoplasticity) voidRatio = void_ratio[p1];
      // const typecode pp1=CODE_GetTypeValue(code[p1]); //phase information

      //Strain rate tensor
      tsymatrix3f D_tensorp1={0,0,0,0,0,0};
      //spin rate tensor
      tfloat3 W_tensorp1={0,0,0};
      //obtain strain and spin tensor
      GetStrainRateSpinTensors_tsym(gradvelp1,D_tensorp1, W_tensorp1);

      // float voidRatio =PhaseHypo[pp1].Hypo_rhoparticle/velrhop1.w -1;  
      // StPhaseHypo hypo_parameters = PhaseHypo[pp1];
      if (tvisco == VISCO_Hypoplasticity){
        const float Hypo_angle = PhaseHypo[pp1].Hypo_angle; ///<  Internal friction angle
        const float Hypo_hs= PhaseHypo[pp1].Hypo_hs;    ///< hs
        const float Hypo_n= PhaseHypo[pp1].Hypo_n;    ///< n
        const float Hypo_ed0= PhaseHypo[pp1].Hypo_ed0;    ///< ed_0
        const float Hypo_ec0= PhaseHypo[pp1].Hypo_ec0;    ///< ec_0
        const float Hypo_ei0= PhaseHypo[pp1].Hypo_ei0;    ///< ei_0 
        const float Hypo_alpha= PhaseHypo[pp1].Hypo_alpha;    ///< alpha
        const float Hypo_beta= PhaseHypo[pp1].Hypo_beta;    ///< beta

        GetStressTensor_tsym_hypo(D_tensorp1, W_tensorp1, SigmaOld_vectorp1, voidRatio, dt, tau_tensorp1,
        Hypo_angle, Hypo_hs, Hypo_n, Hypo_ed0, Hypo_ec0, Hypo_ei0, Hypo_alpha, Hypo_beta,p1);
      }else if (tvisco == VISCO_Elasticity){
        const float lameparm1 = PhaseElastic[pp1].lameparm1;
        const float lameparm2 = PhaseElastic[pp1].lameparm2;
        GetStressTensor_tsym_elastic(D_tensorp1, W_tensorp1, SigmaOld_vectorp1, dt, tau_tensorp1, lameparm2, lameparm1,p1);
      }else{
        //setup stress related variables
        const tsymatrix3f D_tensorp1=D_tensor[p1];
        const float visco_etap1=visco_eta[p1];
        float tau_tensor_magn=0.f;
        float I_t,II_t=0.f; float J1_t,J2_t=0.f;
        GetStressTensor_sym(D_tensorp1,visco_etap1,I_t,II_t,J1_t,J2_t,tau_tensor_magn,tau_tensorp1);
      }
       ////-Sum results together. | Almacena resultados.
      if(tvisco!=VISCO_Artificial) {
        tau[p1].xx=tau_tensorp1.xx; tau[p1].xy=tau_tensorp1.xy; tau[p1].xz=tau_tensorp1.xz;
        tau[p1].yy=tau_tensorp1.yy; tau[p1].yz=tau_tensorp1.yz;
        tau[p1].zz=tau_tensorp1.zz;
        if(tvisco == VISCO_Hypoplasticity) void_ratio[p1] =  voidRatio;
      }
    }
  }
 // Log->Printf("tau of 11163 is: %f %f %f %f %f %f",tau[11163].xx,tau[11163].xy,tau[11163].xz,tau[11163].yy,tau[11163].yz,tau[11163].zz);
      
}


//==============================================================================
/// Perform calculations for NN using the SPH approach: Calculates the effective viscosity
//==============================================================================
template<TpFtMode ftmode,TpVisco tvisco> void JSphCpu::InteractionForcesFluid_NN_SPH_Visco_eta
(unsigned n,unsigned pinit,float visco,float *visco_eta,const tfloat4 *velrhop
  ,const tmatrix3f* gradvel,tsymatrix3f* D_tensor, tfloat3* W_tensor, float *auxnn,float &viscetadt
  ,const tdouble3 *pos,const typecode *code,const unsigned *idp)const
{
  //-Initialize viscth to calculate viscdt maximo con OpenMP. | Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscetath[OMP_MAXTHREADS*OMP_STRIDE];
  for(int th=0; th<OmpThreads; th++)viscetath[th*OMP_STRIDE]=0;
  //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
  const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  for(int p1=int(pinit); p1<pfin; p1++) {

    //-Obtain data of particle p1.
    const tmatrix3f gradvelp1=gradvel[p1];
    const tdouble3 posp1=pos[p1];
    const float rhopp1=velrhop[p1].w;
    //const tsymatrix3f taup1 = (tvisco == VISCO_Artificial ? gradvelp1 : tau[p1]);
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    const typecode pp1=CODE_GetTypeValue(code[p1]); //<vs_non-Newtonian>

    if ((tvisco == VISCO_Hypoplasticity) || (tvisco == VISCO_Elasticity)){
      //Strain rate tensor 
      tsymatrix3f D_tensorp1={0,0,0,0,0,0};
      //spin rate tensor
      tfloat3 W_tensorp1={0,0,0};

      GetStrainRateSpinTensors_tsym(gradvelp1, D_tensorp1, W_tensorp1);
      D_tensor[p1].xx=D_tensorp1.xx; D_tensor[p1].xy=D_tensorp1.xy; D_tensor[p1].xz=D_tensorp1.xz;
      D_tensor[p1].yy=D_tensorp1.yy; D_tensor[p1].yz=D_tensorp1.yz;
      D_tensor[p1].zz=D_tensorp1.zz;

      W_tensor[p1].x = W_tensorp1.x;
      W_tensor[p1].y = W_tensorp1.y;
      W_tensor[p1].z = W_tensorp1.z;

    }else{
      //Strain rate tensor 
      tsymatrix3f D_tensorp1={0,0,0,0,0,0};
      float div_D_tensor=0; float D_tensor_magn=0;
      float I_D,II_D=0.f; float J1_D,J2_D=0.f;
      GetStrainRateTensor_tsym(gradvelp1,I_D,II_D,J1_D,J2_D,div_D_tensor,D_tensor_magn,D_tensorp1);

      //Effective viscosity
      float visco_etap1=0.f;
      float m_NN=PhaseCte[pp1].m_NN; float n_NN=PhaseCte[pp1].n_NN; float tau_yield=PhaseCte[pp1].tau_yield; float visco_NN=PhaseCte[pp1].visco;
      GetEta_Effective(pp1,tau_yield,D_tensor_magn,visco_NN,m_NN,n_NN,visco_etap1);

      //-Sum results together. | Almacena resultados.
      if(visco_etap1) {
        visco_eta[p1]=visco_etap1;
      }
      if(tvisco!=VISCO_Artificial) {
        D_tensor[p1].xx=D_tensorp1.xx; D_tensor[p1].xy=D_tensorp1.xy; D_tensor[p1].xz=D_tensorp1.xz;
        D_tensor[p1].yy=D_tensorp1.yy; D_tensor[p1].yz=D_tensorp1.yz;
        D_tensor[p1].zz=D_tensorp1.zz;
      }
      //debug
      //if(D_tensor_magn){
      //  auxnn[p1] = visco_etap1; // d_tensor_magn;
      //} 
      const int th=omp_get_thread_num();
      if(visco_etap1>viscetath[th*OMP_STRIDE])viscetath[th*OMP_STRIDE]=visco_etap1;
    }
  }
  // need to check the viscosity here
  for(int th=0; th<OmpThreads; th++)if(viscetadt<viscetath[th*OMP_STRIDE])viscetadt=viscetath[th*OMP_STRIDE];
}

//==============================================================================
/// Perform interaction between particles for the SPH approcach using the Const. Eq.: Fluid/Float-Fluid/Float or Fluid/Float-Bound 
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void JSphCpu::InteractionForcesFluid_NN_SPH_ConsEq(unsigned n,unsigned pinit,bool boundp2,float visco
  ,StDivDataCpu divdata,const unsigned *dcell
  ,float *visco_eta,const tsymatrix3f* tau,float *auxnn
  ,const tdouble3 *pos,tfloat4 *velrhop,const typecode *code,const unsigned *idp
  ,tfloat3 *ace,tfloat3 *boundnormalc,bool *BoundCornerc, double &dt)const
{
  //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
    const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif

  for(int p1=int(pinit); p1<pfin; p1++) {
    tfloat3 acep1=TFloat3(0);
    float weight_sum = 0;
    float visc=0;
    //-Obtain data of particle p1 in case of floating objects. | Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1=false;     //-Indicate if it is floating. | Indica si es floating.
    if(USE_FLOATING) {
      ftp1=CODE_IsFloating(code[p1]);
    }

    //-Obtain data of particle p1.
    const tdouble3 posp1=pos[p1];
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const float rhopp1=velrhop[p1].w;
    // if(tvisco!=VISCO_Hypoplasticity) 
    const float visco_etap1=visco_eta[p1]; //<vs_non-Newtonian>
    const tsymatrix3f tau_tensorp1=tau[p1];
    const typecode pp1=(CODE_GetTypeValue(code[p1])); //<vs_non-Newtonian>
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    float Hypo_miu, Elastic_miu;
    if(tvisco==VISCO_Hypoplasticity) Hypo_miu = PhaseHypo[pp1].Hypo_wallfriction;
    if(tvisco==VISCO_Elasticity) Elastic_miu = PhaseElastic[pp1].Elastic_wallfriction;
    float massp1;
    if(tvisco==VISCO_Hypoplasticity){
      massp1=PhaseHypo[pp1].mass;
    }else if(tvisco==VISCO_Elasticity){
      massp1=PhaseElastic[pp1].mass;
    }else{
      massp1=PhaseArray[pp1].mass; 
    }

    //-Search for neighbours in adjacent cells.
    const StNgSearch ngs=nsearch::Init(dcell[p1],boundp2,divdata);
    for(int z=ngs.zini; z<ngs.zfin; z++)for(int y=ngs.yini; y<ngs.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs,divdata);

      //-Interaction of Fluid with type Fluid or Bound. | Interaccion de Fluid con varias Fluid o Bound.
      //------------------------------------------------------------------------------------------------
      bool rsym=false; //<vs_syymmetry>
      for(unsigned p2=pif.x; p2<pif.y; p2++) {
        const float drx=float(posp1.x-pos[p2].x);
        float dry=float(posp1.y-pos[p2].y);
        if(rsym)   dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=KernelSize2 && rr2>=ALMOSTZERO) {
          //-Computes kernel.
          const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
          const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

          //===== Get mass of particle p2 ===== 
          //<vs_non-Newtonian>
          const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(code[p2])); //<vs_non-Newtonian>
          float massp2;
          if(tvisco==VISCO_Hypoplasticity){
            massp2=(boundp2 ? MassBound : PhaseHypo[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
            //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);
          }else if(tvisco==VISCO_Elasticity){
            massp2=(boundp2 ? MassBound : PhaseElastic[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
          }else{
            massp2=(boundp2 ? MassBound : PhaseArray[pp2].mass);
          }


          //Floating
          bool ftp2=false;    //-Indicate if it is floating | Indica si es floating.
          bool compute=true;  //-Deactivate when using DEM and if it is of type float-float or float-bound | Se desactiva cuando se usa DEM y es float-float o float-bound.
          if(USE_FLOATING) {
            ftp2=CODE_IsFloating(code[p2]);
            if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
            compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
          }

          tfloat4 velrhop2=velrhop[p2];
          if(rsym)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

          //-velocity dvx.
          const float dvx=velp1.x-velrhop2.x,dvy=velp1.y-velrhop2.y,dvz=velp1.z-velrhop2.z;
          // const float cbar=(tvisco!=VISCO_Hypoplasticity ? max(PhaseArray[pp1].Cs0,PhaseArray[pp2].Cs0) : max(PhaseHypo[pp1].Cs0,PhaseHypo[pp2].Cs0));
          float cbar;
          if (tvisco==VISCO_Hypoplasticity) cbar = PhaseHypo[pp2].Cs0;
          else if (tvisco==VISCO_Elasticity) cbar = PhaseElastic[pp2].Cs0;
          else cbar = PhaseArray[pp2].Cs0;
          //===== Viscosity ===== 
          if(compute) {
            const float dot=drx*dvx+dry*dvy+drz*dvz;
            const float dot_rr2=dot/(rr2+Eta2);
            visc=max(dot_rr2,visc);

            tsymatrix3f tau_tensorp2=tau[p2]; tsymatrix3f tau_sum={0,0,0,0,0,0};
            if(boundp2 & (tvisco != VISCO_Hypoplasticity) & (tvisco != VISCO_Elasticity)){
              tau_tensorp2=tau_tensorp1;
            }
            if ((tvisco == VISCO_Hypoplasticity) || (tvisco == VISCO_Elasticity)){
              if(boundp2 & SlipMode == SLIP_Friction_Soleimani){
                tfloat3 normal = boundnormalc[p2];
                 //Log->Printf("normal = %f %f %f", normal.x, normal.y,  normal.z);
                const float normal_mag = sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);
                if (normal_mag!=0){
                  normal.x = normal.x/normal_mag;
                  normal.y = normal.y/normal_mag;
                  normal.z = normal.z/normal_mag;
                }
                // Log->Printf("ace = %f %f %f", acep1.x, acep1.y ,acep1.z);
                
                tfloat3 tau_vector = TFloat3(0);
                tau_vector.x = normal.x*tau_tensorp2.xx + normal.y*tau_tensorp2.xy + normal.z*tau_tensorp2.xz;
                tau_vector.y = normal.x*tau_tensorp2.xy + normal.y*tau_tensorp2.yy + normal.z*tau_tensorp2.yz;
                tau_vector.z = normal.x*tau_tensorp2.xz + normal.y*tau_tensorp2.yz + normal.z*tau_tensorp2.zz;
                float t_n = abs(tau_vector.x*normal.x + tau_vector.y*normal.y + tau_vector.z*normal.z);
                tfloat3 t_tau_vector = TFloat3(0);
                t_tau_vector.x = tau_vector.x - t_n*normal.x;
                t_tau_vector.y = tau_vector.y - t_n*normal.y;
                t_tau_vector.z = tau_vector.z - t_n*normal.z;
                float t_s = sqrt(t_tau_vector.x*t_tau_vector.x+t_tau_vector.y*t_tau_vector.y+t_tau_vector.z*t_tau_vector.z);
                float miu = (VISCO_Hypoplasticity ? Hypo_miu : Elastic_miu);
                  if (t_s>miu*t_n){
                    t_s = miu*t_n;
                  }
                const float temp = -dvx*normal.x - dvy*normal.y - dvz*normal.z;
                tfloat3 dv_normal_vector = {temp*normal.x,temp*normal.y,temp*normal.z};
                tfloat3 dv_tangent_vector = TFloat3(0); 
                dv_tangent_vector.x= -dvx - dv_normal_vector.x;
                dv_tangent_vector.y= -dvy - dv_normal_vector.y;
                dv_tangent_vector.z= -dvz - dv_normal_vector.z;
                const float dv_tangent_magnitude  = sqrt(dv_tangent_vector.x*dv_tangent_vector.x+dv_tangent_vector.y*dv_tangent_vector.y+dv_tangent_vector.z*dv_tangent_vector.z);
                tfloat3 tangent = TFloat3(0);
                if (dv_tangent_magnitude!=0){
                  tangent = {dv_tangent_vector.x/dv_tangent_magnitude, dv_tangent_vector.y/dv_tangent_magnitude, dv_tangent_vector.z/dv_tangent_magnitude};
                }
                //store acceleration
                acep1.x+= (t_n*normal.x+t_s*tangent.x)*fac;
                acep1.y+= (t_n*normal.y+t_s*tangent.y)*fac;
                acep1.z+= (t_n*normal.z+t_s*tangent.z)*fac;
                weight_sum+= fac;
                //if(p1==11163){Log->Printf("ace of particle 11163 at soil-bound interaction is: %d %f %f %f",p2,acep1.x,acep1.y,acep1.z);
                //Log->Printf("p2 and tau_tensorp2 are: %d %f %f %f %f %f %f", p2,tau_tensorp2.xx,tau_tensorp2.xy,tau_tensorp2.xz,tau_tensorp2.yy,tau_tensorp2.yz,tau_tensorp2.zz);
                //Log->Printf("tn t_s normal tangent weight_sum %f %f %f %f %f %f %f %f %f",t_n,t_s,normal.x, normal.y, normal.z, tangent.x, tangent.y,tangent.z,weight_sum);
                //}
              }else{ // The thrid boundary setting for granular material. Non-frictional. And for fluid-fluid interaction. 
                float taux, tauy, tauz;
                tau_sum.xx=tau_tensorp1.xx/pow(rhopp1,2) + tau_tensorp2.xx/pow(velrhop2.w,2);
                tau_sum.xy=tau_tensorp1.xy/pow(rhopp1,2) + tau_tensorp2.xy/pow(velrhop2.w,2);
                tau_sum.xz=tau_tensorp1.xz/pow(rhopp1,2) + tau_tensorp2.xz/pow(velrhop2.w,2);
                tau_sum.yy=tau_tensorp1.yy/pow(rhopp1,2)+tau_tensorp2.yy/pow(velrhop2.w,2);
                tau_sum.yz=tau_tensorp1.yz/pow(rhopp1,2)+tau_tensorp2.yz/pow(velrhop2.w,2);
                tau_sum.zz=tau_tensorp1.zz/pow(rhopp1,2)+tau_tensorp2.zz/pow(velrhop2.w,2);
                taux=(tau_sum.xx*frx+tau_sum.xy*fry+tau_sum.xz*frz); // as per symetric tensor grad
                tauy=(tau_sum.xy*frx+tau_sum.yy*fry+tau_sum.yz*frz);
                tauz=(tau_sum.xz*frx+tau_sum.yz*fry+tau_sum.zz*frz);
                //store acceleration
                acep1.x+=taux*massp2; acep1.y+=tauy*massp2; acep1.z+=tauz*massp2;
                // if(p1==11163)Log->Printf("ace of particle 11163 at soil-soil interaction is: %d %f %f %f",p2,acep1.x,acep1.y,acep1.z);

              }
            }else{
              float taux, tauy, tauz;
              tau_sum.xx=tau_tensorp1.xx+tau_tensorp2.xx;	tau_sum.xy=tau_tensorp1.xy+tau_tensorp2.xy;	tau_sum.xz=tau_tensorp1.xz+tau_tensorp2.xz;
              tau_sum.yy=tau_tensorp1.yy+tau_tensorp2.yy;	tau_sum.yz=tau_tensorp1.yz+tau_tensorp2.yz;
              tau_sum.zz=tau_tensorp1.zz+tau_tensorp2.zz;

              taux=(tau_sum.xx*frx+tau_sum.xy*fry+tau_sum.xz*frz)/(velrhop2.w); // as per symetric tensor grad
              tauy=(tau_sum.xy*frx+tau_sum.yy*fry+tau_sum.yz*frz)/(velrhop2.w);
              tauz=(tau_sum.xz*frx+tau_sum.yz*fry+tau_sum.zz*frz)/(velrhop2.w);
              //store acceleration
              acep1.x+=taux*massp2; acep1.y+=tauy*massp2; acep1.z+=tauz*massp2;
            }
          }
          //-SPS turbulence model.
          //-SPS turbulence model is disabled in v5.0 NN version
          rsym=(rsymp1&&!rsym && float(posp1.y-dry)<=KernelSize); //<vs_syymmetry>
          if(rsym)p2--;																									//<vs_syymmetry>
        }
        else rsym=false;																								//<vs_syymmetry>		
      }
    }
    //-Sum results together. | Almacena resultados.
    if(acep1.x||acep1.y||acep1.z) {
      if ((tvisco==VISCO_Hypoplasticity || tvisco==VISCO_Elasticity) &&  (SlipMode==SLIP_Friction_Soleimani ) && boundp2){
        acep1.x = acep1.x*pow(Dp, 2)/massp1/weight_sum;
        acep1.y = acep1.y*pow(Dp, 2)/massp1/weight_sum;
        acep1.z = acep1.z*pow(Dp, 2)/massp1/weight_sum;
      }

      ace[p1]=ace[p1]+acep1;
    }
  }
 //Log->Printf("ace of particle 11163 after conseq interaction is: %f %f %f",ace[11163].x,ace[11163].y,ace[11163].z);
}

//==============================================================================
/// Perform interaction between particles for the SPH approcach using the Const. Eq.: 
/// Fluid/Float-Bound, particullarly for Zhan's method 
//==============================================================================

template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void JSphCpu::InteractionForcesFluid_NN_SPH_ConsEq_Zhan_bound(unsigned n,unsigned pinit,bool boundp2,float visco
  ,StDivDataCpu divdata,const unsigned *dcell
  ,float *visco_eta,const tsymatrix3f* tau,tfloat3* Force,float *auxnn
  ,const tdouble3 *pos,tfloat4 *velrhop,const typecode *code,const unsigned *idp
  ,tfloat3 *ace,tfloat3 *boundnormalc,bool *BoundCornerc, double &dt)const
{
  //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
   // Log->Printf("Goes into f-b interaction only");
    const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  
  for(int p1=int(pinit); p1<pfin; p1++) {
    tfloat3 acep1=TFloat3(0);
    tfloat3 Forcep2=TFloat3(0);
    float weight_sum = 0;
    float visc=0;
    //-Obtain data of particle p1 in case of floating objects. | Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1=false;     //-Indicate if it is floating. | Indica si es floating.
    if(USE_FLOATING) {
      ftp1=CODE_IsFloating(code[p1]);
    }

    //-Obtain data of particle p1.
    const tdouble3 posp1=pos[p1];
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const float rhopp1=velrhop[p1].w;
    // if(tvisco!=VISCO_Hypoplasticity) 
    const float visco_etap1=visco_eta[p1]; //<vs_non-Newtonian>
    const tsymatrix3f tau_tensorp1=tau[p1];
    const typecode pp1=(CODE_GetTypeValue(code[p1])); //<vs_non-Newtonian>
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    float Hypo_miu, Elastic_miu;
    if(tvisco==VISCO_Hypoplasticity) Hypo_miu = PhaseHypo[pp1].Hypo_wallfriction;
    if(tvisco==VISCO_Elasticity) Elastic_miu = PhaseElastic[pp1].Elastic_wallfriction;
    float massp1;
    if(tvisco==VISCO_Hypoplasticity){
      massp1=PhaseHypo[pp1].mass;
    }else if(tvisco==VISCO_Elasticity){
      massp1=PhaseElastic[pp1].mass;
    }else{
      massp1=PhaseArray[pp1].mass; 
    }

    //-Search for neighbours in adjacent cells.
    const StNgSearch ngs=nsearch::Init(dcell[p1],boundp2,divdata);
    float rr2; float drx; float dry; float drz; int p2; bool rsym=false; //<vs_syymmetry>

    int p2_nearest = 0; // p2_nearest: nearest boundary particle to p1
    drx=float(posp1.x - pos[p2_nearest].x);
    dry=float(posp1.y - pos[p2_nearest].y);
    if(rsym)  dry=float(posp1.y + pos[p2_nearest].y); //<vs_syymmetry>
    drz=float(posp1.z - pos[p2_nearest].z);
    rr2=drx*drx+dry*dry+drz*drz;
    float rr2_temp=rr2;

    for(int z=ngs.zini; z<ngs.zfin; z++)for(int y=ngs.yini; y<ngs.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs,divdata);
      // Find nearest boundary particle p2 with regard to material particle p1
      for(unsigned p2=pif.x ; p2<pif.y; p2++) {
        drx=float(posp1.x-pos[p2].x);
        dry=float(posp1.y-pos[p2].y);
        if(rsym)   dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        float drz=float(posp1.z-pos[p2].z);
        float rr2_temp = drx*drx+dry*dry+drz*drz;
        if(rr2_temp<rr2){ 
          p2_nearest = p2; 
          rr2 = rr2_temp;
          //Log->Printf("p2 nearest is: %d",p2_nearest);
          //Log->Printf("rr2 is: %f:", rr2);
        }
      } 
    }
    p2 = p2_nearest;
    drx=float(posp1.x - pos[p2].x);
    dry=float(posp1.y - pos[p2].y);
    if(rsym)  dry=float(posp1.y + pos[p2].y); //<vs_syymmetry>
    drz=float(posp1.z - pos[p2].z);  
  //  if(p1==62) Log->Printf("p2 is: %d",p2);  
      //-Interaction of Fluid with type Fluid or Bound. | Interaccion de Fluid con varias Fluid o Bound.
      //------------------------------------------------------------------------------------------------
    if(rr2<=KernelSize2 && rr2>=ALMOSTZERO) {
      //-Computes kernel.
      const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //===== Get mass of particle p2 ===== 
      //<vs_non-Newtonian>
      const typecode pp2=pp1; //<vs_non-Newtonian>
      float massp2 = MassBound;
      
      //Floating
      bool ftp2=false;    //-Indicate if it is floating | Indica si es floating.
      bool compute=true;  //-Deactivate when using DEM and if it is of type float-float or float-bound | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING) {
        ftp2=CODE_IsFloating(code[p2]);
        if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
        compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      }

      tfloat4 velrhop2=velrhop[p2];
      if(rsym)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

      //-velocity dvx.
      const float dvx=velp1.x-velrhop2.x,dvy=velp1.y-velrhop2.y,dvz=velp1.z-velrhop2.z;
      // const float cbar=(tvisco!=VISCO_Hypoplasticity ? max(PhaseArray[pp1].Cs0,PhaseArray[pp2].Cs0) : max(PhaseHypo[pp1].Cs0,PhaseHypo[pp2].Cs0));
      float cbar;
      if (tvisco==VISCO_Hypoplasticity) cbar = PhaseHypo[pp2].Cs0;
      else if (tvisco==VISCO_Elasticity) cbar = PhaseElastic[pp2].Cs0;
      else cbar = PhaseArray[pp2].Cs0;
      //===== Viscosity ===== 
      if(compute) {
        const float dot=drx*dvx+dry*dvy+drz*dvz;
        const float dot_rr2=dot/(rr2+Eta2);
        visc=max(dot_rr2,visc);

        tsymatrix3f tau_tensorp2=tau[p2]; tsymatrix3f tau_sum={0,0,0,0,0,0};
        if(boundp2 & (tvisco != VISCO_Hypoplasticity) & (tvisco != VISCO_Elasticity)){
          tau_tensorp2=tau_tensorp1;
        }
        if ((tvisco == VISCO_Hypoplasticity) || (tvisco == VISCO_Elasticity)){
          tfloat3 normal;
          double ds;
          float miu = (tvisco == VISCO_Hypoplasticity ? Hypo_miu : Elastic_miu);
          if(BoundCornerc[p2]){ // Redefine normal for corner bound particles
            ds = sqrt(rr2);
            if(ds!=0) normal.x = drx/ds; normal.y = dry/ds; normal.z = drz/ds;
          }
          else{
            normal = boundnormalc[p2];  
            // Log->Printf("normal = %f %f %f", normal.x, normal.y,  normal.z);
            const float normal_mag = sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);
            if (normal_mag!=0){
            normal.x = normal.x/normal_mag;
            normal.y = normal.y/normal_mag;
            normal.z = normal.z/normal_mag;
            }
            ds = abs(drx*normal.x + dry*normal.y + drz*normal.z);
          }
          // Check any fluid particle p1 within dp of bound particle p2
          if((ds < Dp) & ((dvx*normal.x + dvy*normal.y + dvz*normal.z ) < 0)){ 
            //// need to check latter what is velocity_predicted.
            // Normal contact force magnitude from boundary to soil particle
            //float t_n_mag = abs((normal.x*dvx + normal.y*dvy + normal.z*dvz)/(dt/massp2 + dt/massp1)); //This is when p2 is floating boundary
            float t_n_mag = abs((normal.x*dvx + normal.y*dvy + normal.z*dvz)/(dt/massp1));
            tfloat3 t_n; 
            t_n.x = t_n_mag*normal.x; t_n.y = t_n_mag*normal.y; t_n.z = t_n_mag*normal.z;
            // Shear contact force magnitude from boundary to soil particles
            tfloat3 t_s;
            //t_s.x = -dvx/(dt/massp2 + dt/massp1) - t_n.x; This is when p2 is floating boundary
            //t_s.y = -dvy/(dt/massp2 + dt/massp1) - t_n.y;
            //t_s.z = -dvz/(dt/massp2 + dt/massp1) - t_n.z;
            if(SlipMode == SLIP_FreeSlip_Zhan){
              t_s.x = 0; t_s.y=0; t_s.z=0;
            }else{ // frictional slip or no-slip
            t_s.x = -dvx/(dt/massp1) - t_n.x;
            t_s.y = -dvy/(dt/massp1) - t_n.y;
            t_s.z = -dvz/(dt/massp1) - t_n.z;
            }
            float t_s_mag = sqrt(t_s.x*t_s.x + t_s.y*t_s.y + t_s.z*t_s.z);
            if((t_s_mag >= miu*t_n_mag) && (t_s_mag!=0) && SlipMode == SLIP_Friction_Zhan){
              t_s.x = miu*t_n_mag/t_s_mag*t_s.x;
              t_s.y = miu*t_n_mag/t_s_mag*t_s.y;
              t_s.z = miu*t_n_mag/t_s_mag*t_s.z;
              t_s_mag = sqrt(t_s.x*t_s.x + t_s.y*t_s.y + t_s.z*t_s.z);
            }
            //store acceleration
            acep1.x= t_n.x + t_s.x;
            acep1.y= t_n.y + t_s.y;
            acep1.z= t_n.z + t_s.z;
            Forcep2.x = -acep1.x*massp2;
            Forcep2.y = -acep1.y*massp2;
            Forcep2.z = -acep1.z*massp2;
          }   
        }
      }
      rsym=(rsymp1&&!rsym && float(posp1.y-dry)<=KernelSize); //<vs_syymmetry>
    if(rsym)p2--;																	  					//<vs_syymmetry>
    }
    else rsym=false;																		  			//<vs_syymmetry>		
  //-Sum results together. | Almacena resultados.
  if(acep1.x||acep1.y||acep1.z) {
    if ((tvisco==VISCO_Hypoplasticity || tvisco==VISCO_Elasticity)){
      // Update velocity to n+1/2 step, not sure wether to include weight_sum:  acep1.x=acep1.x/massp1/weight_sum??
      acep1.x = acep1.x/massp1;
      acep1.y = acep1.y/massp1;
      acep1.z = acep1.z/massp1;
      velrhop[p1].x += dt*acep1.x; //From predicted to corrected velocity at n+1/2 step.
      velrhop[p1].y += dt*acep1.y;
      velrhop[p1].z += dt*acep1.z;
    }
    ace[p1]=ace[p1]+acep1;
    Force[p2]=Force[p2]+Forcep2;
  }
}
}
//==============================================================================
/// Perform interaction between particles for the SPH approcach using the Morris operator: Fluid/Float-Fluid/Float or Fluid/Float-Bound 
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void JSphCpu::InteractionForcesFluid_NN_SPH_Morris(unsigned n,unsigned pinit,bool boundp2,float visco
  ,StDivDataCpu divdata,const unsigned *dcell
  ,float *visco_eta,const tsymatrix3f* tau,tmatrix3f* gradvel,float *auxnn
  ,const tdouble3 *pos,const tfloat4 *velrhop,const typecode *code,const unsigned *idp
  ,tfloat3 *ace)const
{
  //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
  const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  for(int p1=int(pinit); p1<pfin; p1++) {
    tfloat3 acep1=TFloat3(0);
    float visc=0;
    //-Obtain data of particle p1 in case of floating objects. | Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1=false;     //-Indicate if it is floating. | Indica si es floating.
    if(USE_FLOATING) {
      ftp1=CODE_IsFloating(code[p1]);
    }

    //-Obtain data of particle p1.
    const tdouble3 posp1=pos[p1];
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const float rhopp1=velrhop[p1].w;
    // if(tvisco!=VISCO_Hypoplasticity) 
    const float visco_etap1=visco_eta[p1];
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>
    const typecode pp1=CODE_GetTypeValue(code[p1]); //<vs_non-Newtonian>

                                                      //-Search for neighbours in adjacent cells.
    const StNgSearch ngs=nsearch::Init(dcell[p1],boundp2,divdata);
    for(int z=ngs.zini; z<ngs.zfin; z++)for(int y=ngs.yini; y<ngs.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs,divdata);

      //-Interaction of Fluid with type Fluid or Bound. | Interaccion de Fluid con varias Fluid o Bound.
      //------------------------------------------------------------------------------------------------
      bool rsym=false; //<vs_syymmetry>
      for(unsigned p2=pif.x; p2<pif.y; p2++) {
        const float drx=float(posp1.x-pos[p2].x);
        float dry=float(posp1.y-pos[p2].y);
        if(rsym)    dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=KernelSize2 && rr2>=ALMOSTZERO){
          //-Computes kernel.
          const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
          const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

          //===== Get mass of particle p2 ===== 
          //<vs_non-Newtonian>
          const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(code[p2])); //<vs_non-Newtonian>
          float massp2;
          if(tvisco==VISCO_Hypoplasticity){
            massp2=(boundp2 ? MassBound : PhaseHypo[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
            //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);
          }else if(tvisco==VISCO_Elasticity){
            massp2=(boundp2 ? MassBound : PhaseElastic[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
          }else{massp2=(boundp2 ? MassBound : PhaseArray[pp2].mass); }
          // float massp2=(boundp2 ? MassBound : PhaseArray[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
          //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);

          //Floating
          bool ftp2=false;    //-Indicate if it is floating | Indica si es floating.
          bool compute=true;  //-Deactivate when using DEM and if it is of type float-float or float-bound | Se desactiva cuando se usa DEM y es float-float o float-bound.
          if(USE_FLOATING) {
            ftp2=CODE_IsFloating(code[p2]);
            if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
            compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
          }

          tfloat4 velrhop2=velrhop[p2];
          if(rsym)velrhop2.y=-velrhop2.y; //<vs_syymmetry>                                               
          //-velocity dvx.
          float dvx=velp1.x-velrhop2.x,dvy=velp1.y-velrhop2.y,dvz=velp1.z-velrhop2.z;
          if(boundp2) { //this applies no slip on tensor                                     
            dvx=2.f*velp1.x; dvy=2.f*velp1.y; dvz=2.f*velp1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
          }
          // const float cbar=(tvisco!=VISCO_Hypoplasticity ? max(PhaseArray[pp1].Cs0,PhaseArray[pp2].Cs0) : max(PhaseHypo[pp1].Cs0,PhaseHypo[pp2].Cs0));
          float cbar;
          if(tvisco==VISCO_Hypoplasticity) {cbar=PhaseHypo[pp2].Cs0;
          }else if(tvisco==VISCO_Elasticity) {cbar=PhaseElastic[pp2].Cs0;
          }else{cbar=PhaseArray[pp2].Cs0;}
          // const float cbar=max(PhaseArray[pp2].Cs0,PhaseArray[pp2].Cs0);

          //===== Viscosity ===== 
          if(compute) {
            const float dot=drx*dvx+dry*dvy+drz*dvz;
            const float dot_rr2=dot/(rr2+Eta2);
            visc=max(dot_rr2,visc);
            float visco_NN=PhaseCte[pp2].visco;
            if(tvisco==VISCO_Hypoplasticity) {visco_NN=PhaseHypo[pp2].visco;
            }else if(tvisco==VISCO_Elasticity) {visco_NN=PhaseElastic[pp2].visco;
            }
            
            if((tvisco==VISCO_Artificial)|| (tvisco== VISCO_Hypoplasticity)|| (tvisco== VISCO_Elasticity) ) {//-Artificial viscosity.
              if(dot<0) {
                const float amubar=KernelH*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
                const float robar=(rhopp1+velrhop2.w)*0.5f;
                const float pi_visc=(-visco_NN*cbar*amubar/robar)*massp2;
                acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
              }
            }
            else if(tvisco==VISCO_LaminarSPS) {//-Laminar viscosity.
              {//-Laminar contribution.
                float visco_etap2=visco_eta[p2];
                //Morris Operator
                if(boundp2)visco_etap2=visco_etap1;
                const float temp=(visco_etap1+visco_etap2)/((rr2+Eta2)*velrhop2.w);
                const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz);
                acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
              }
              //-SPS turbulence model.
              //-SPS turbulence model is disabled in v5.0 NN version
            }
          }
          rsym=(rsymp1&&!rsym && float(posp1.y-dry)<=KernelSize); //<vs_syymmetry>
          if(rsym)p2--;                                       //<vs_syymmetry>
        }
        else rsym=false; //<vs_syymmetry>
      }
    }
    //-Sum results together. | Almacena resultados.
    if(acep1.x||acep1.y||acep1.z) {
      ace[p1]=ace[p1]+acep1;
    }
  }
}

//==============================================================================
/// Perform interaction between particles for the SPH approcach, Pressure and vel gradients: Fluid/Float-Fluid/Float or Fluid/Float-Bound 
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void JSphCpu::InteractionForcesFluid_NN_SPH_PressGrad(unsigned n,unsigned pinit,bool boundp2,float visco
  ,StDivDataCpu divdata,const unsigned *dcell
  ,tmatrix3f* gradvel
  ,const tdouble3 *pos,const tfloat4 *velrhop,const typecode *code,const unsigned *idp
  ,const float *press
  ,float &viscdt,float *ar,tfloat3 *ace,float *delta
  ,TpShifting shiftmode,tfloat4 *shiftposfs,int Zhan_loop)const
{
  //-Initialize viscth to calculate viscdt maximo con OpenMP. | Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscth[OMP_MAXTHREADS*OMP_STRIDE];
  for(int th=0; th<OmpThreads; th++)viscth[th*OMP_STRIDE]=0;
  //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
  const int pfin=int(pinit+n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
  for(int p1=int(pinit); p1<pfin; p1++) {
      
    float visc=0,arp1=0,deltap1=0;
    tfloat3 acep1=TFloat3(0);
    tmatrix3f gradvelp1={0,0,0,0,0,0,0,0,0};
    //-Variables for Shifting.
    tfloat4 shiftposfsp1;
    if(shift)shiftposfsp1=shiftposfs[p1];

    //-Obtain data of particle p1 in case of floating objects. | Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1=false;     //-Indicate if it is floating. | Indica si es floating.
    if(USE_FLOATING) {
      ftp1=CODE_IsFloating(code[p1]);
      if(ftp1 && tdensity!=DDT_None)deltap1=FLT_MAX; //-DDT is not applied to floating particles.
      if(ftp1 && shift)shiftposfsp1.x=FLT_MAX;  //-For floating objects do not calculate shifting. | Para floatings no se calcula shifting.
    }

    //-Obtain data of particle p1.
    const tdouble3 posp1=pos[p1];
    // Log->Print("mark JSphCpu NN SPH strain grad -0 \n");
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const float rhopp1=velrhop[p1].w;
    const float pressp1=press[p1];
    // Log->Print("p=%u, pressure=%f",p1, pressp1);
    // Log->Print("mark JSphCpu NN SPH strain grad -1 \n");
    const bool rsymp1=(Symmetry && posp1.y<=KernelSize); //<vs_syymmetry>                                 
    //<vs_non-Newtonian>
    const typecode pp1=CODE_GetTypeValue(code[p1]);

    //-Search for neighbours in adjacent cells.
    const StNgSearch ngs=nsearch::Init(dcell[p1],boundp2,divdata);
    for(int z=ngs.zini; z<ngs.zfin; z++)for(int y=ngs.yini; y<ngs.yfin; y++) {
      const tuint2 pif=nsearch::ParticleRange(y,z,ngs,divdata);

      //-Interaction of Fluid with type Fluid or Bound. | Interaccion de Fluid con varias Fluid o Bound.
      //------------------------------------------------------------------------------------------------
      bool rsym=false; //<vs_syymmetry>
      for(unsigned p2=pif.x; p2<pif.y; p2++) {
        const float drx=float(posp1.x-pos[p2].x);
        float dry=float(posp1.y-pos[p2].y);
        if(rsym)    dry=float(posp1.y+pos[p2].y); //<vs_syymmetry>
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=KernelSize2 && rr2>=ALMOSTZERO) {
          //-Computes kernel.
          const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
          const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients

          //===== Get mass of particle p2 ===== 
          //<vs_non-Newtonian>
          const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(code[p2])); //<vs_non-Newtonian>
          float massp2;
          if(tvisco==VISCO_Hypoplasticity){
            massp2=(boundp2 ? MassBound : PhaseHypo[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
            //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);
          }else if(tvisco==VISCO_Elasticity){
            massp2=(boundp2 ? MassBound : PhaseElastic[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
          }else massp2=(boundp2 ? MassBound : PhaseArray[pp2].mass);
          // float massp2=(boundp2 ? MassBound : PhaseArray[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
          // //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);

          //Floating                      
          bool ftp2=false;    //-Indicate if it is floating | Indica si es floating.
          bool compute=true;  //-Deactivate when using DEM and if it is of type float-float or float-bound | Se desactiva cuando se usa DEM y es float-float o float-bound.
          if(USE_FLOATING) {
            ftp2=CODE_IsFloating(code[p2]);
            if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
#ifdef DELTA_HEAVYFLOATING
            if(ftp2 && tdensity==DDT_DDT && massp2<=(MassFluid*1.2f))deltap1=FLT_MAX;
#else
            if(ftp2 && tdensity==DDT_DDT)deltap1=FLT_MAX;
#endif
            if(ftp2 && shift && shiftmode==SHIFT_NoBound)shiftposfsp1.x=FLT_MAX; //-With floating objects do not use shifting. | Con floatings anula shifting.
            compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
          }

          tfloat4 velrhop2=velrhop[p2];
          if(rsym)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

          //===== Acceleration ===== 
          if(compute) {
            if ((tvisco != VISCO_Hypoplasticity) & (tvisco != VISCO_Elasticity)){
              const float prs=(pressp1+press[p2])/(rhopp1*velrhop2.w)+(tker==KERNEL_Cubic ? fsph::GetKernelCubic_Tensil(CSP,rr2,rhopp1,pressp1,velrhop2.w,press[p2]) : 0);
              const float p_vpm=-prs*massp2;
              acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
            }
          }

          //-Density derivative.
          const float rhop1over2=rhopp1/velrhop2.w;
          float dvx=velp1.x-velrhop2.x,dvy=velp1.y-velrhop2.y,dvz=velp1.z-velrhop2.z;
          if(compute)arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz)*rhop1over2;
          // const float cbar=(tvisco!=VISCO_Hypoplasticity ? max(PhaseArray[pp1].Cs0,PhaseArray[pp2].Cs0) : max(PhaseHypo[pp1].Cs0,PhaseHypo[pp2].Cs0));
          float cbar;
          if(tvisco==VISCO_Hypoplasticity) {cbar=PhaseHypo[pp2].Cs0;}
          else if(tvisco==VISCO_Elasticity) {cbar=PhaseElastic[pp2].Cs0;}
          else {cbar=PhaseArray[pp2].Cs0;}
          // const float cbar=max(PhaseArray[pp2].Cs0,PhaseArray[pp2].Cs0); //<vs_non-Newtonian>
          //-Density Diffusion Term (DeltaSPH Molteni).
          if(tdensity==DDT_DDT && deltap1!=FLT_MAX) {
            const float visc_densi=DDTkh*cbar*(rhop1over2-1.f)/(rr2+Eta2);
            const float dot3=(drx*frx+dry*fry+drz*frz);
            const float delta=(pp1==pp2 ? visc_densi*dot3*massp2 : 0); //<vs_non-Newtonian>
            //deltap1 = (boundp2 ? FLT_MAX : deltap1 + delta);
            deltap1=(boundp2 && TBoundary==BC_DBC ? FLT_MAX : deltap1+delta);
          }
          //-Density Diffusion Term (Fourtakas et al 2019).  //<vs_dtt2_ini>
          if((tdensity==DDT_DDT2||(tdensity==DDT_DDT2Full&&!boundp2))&&deltap1!=FLT_MAX&&!ftp2) {
            const float rh=1.f+DDTgz*drz;
            const float drhop=RhopZero*pow(rh,1.f/Gamma)-RhopZero;
            const float visc_densi=DDTkh*cbar*((velrhop2.w-rhopp1)-drhop)/(rr2+Eta2);
            const float dot3=(drx*frx+dry*fry+drz*frz);
            const float delta=(pp1==pp2 ? visc_densi*dot3*massp2/velrhop2.w : 0); //<vs_non-Newtonian>
            deltap1=(boundp2 ? FLT_MAX : deltap1-delta);
          }  //<vs_dtt2_end>

             //-Shifting correction.
          if(shift && shiftposfsp1.x!=FLT_MAX) {
            bool heavyphase;
            if(tvisco==VISCO_Hypoplasticity){
              heavyphase=(PhaseHypo[pp1].mass>PhaseHypo[pp2].mass && pp1!=pp2 ? true : false);
            }else if(tvisco==VISCO_Elasticity){
              heavyphase=(PhaseElastic[pp1].mass>PhaseElastic[pp2].mass && pp1!=pp2 ? true : false);
            }else{
              heavyphase=(PhaseArray[pp1].mass>PhaseArray[pp2].mass && pp1!=pp2 ? true : false);
            }
            const float massrhop=massp2/velrhop2.w;
            const bool noshift=(boundp2&&(shiftmode==SHIFT_NoBound||(shiftmode==SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
            shiftposfsp1.x=(noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x+massrhop*frx)); //-For boundary do not use shifting. | Con boundary anula shifting.                            
            shiftposfsp1.y+=(heavyphase ? 0 : massrhop*fry); //<vs_non-Newtonian>
            shiftposfsp1.z+=(heavyphase ? 0 : massrhop*frz); //<vs_non-Newtonian>
            shiftposfsp1.w-=(heavyphase ? 0 : massrhop*(drx*frx+dry*fry+drz*frz)); //<vs_non-Newtonian>
          }

          //===== Calculate velocity gradient and artificial viscosity  ===== 
          if(compute) {
            const float dot=drx*dvx+dry*dvy+drz*dvz;
            const float dot_rr2=dot/(rr2+Eta2);
            visc=max(dot_rr2,visc);
            if(tvisco!=VISCO_Artificial) { //<vs_non-Newtonian>
              {//vel gradients
                if(boundp2 & (tvisco != VISCO_Hypoplasticity) && (tvisco != VISCO_Elasticity)) {
                  dvx=2.f*velp1.x; dvy=2.f*velp1.y; dvz=2.f*velp1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
                }
                ///////////////////////////////////////
                if(Zhan_loop==0 || !boundp2) {
                
               GetVelocityGradients_SPH_tsym(massp2,velrhop2,dvx,dvy,dvz,frx,fry,frz,gradvelp1);
                }
                //////////////////////////////////////
              }
            }
            if((tvisco== VISCO_Hypoplasticity) || (tvisco== VISCO_Elasticity)) {//-Artificial viscosity.
              // const float cbar= max(PhaseHypo[pp1].Cs0,PhaseHypo[pp2].Cs0);
              float cbar;
              float visco_NN;
              if (tvisco == VISCO_Hypoplasticity){
                cbar = PhaseHypo[pp2].Cs0;
                visco_NN= PhaseHypo[pp2].visco;
              }else{
                cbar = PhaseElastic[pp2].Cs0;
                visco_NN= PhaseElastic[pp2].visco; 
              }
              if(dot<0) {
                const float amubar=KernelH*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
                const float robar=(rhopp1+velrhop2.w)*0.5f;
                const float pi_visc=(-visco_NN*cbar*amubar/robar)*massp2;
                acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
              }
            }
          }
          rsym=(rsymp1&&!rsym && float(posp1.y-dry)<=KernelSize);   //<vs_syymmetry>
          if(rsym)p2--;																										//<vs_syymmetry>
        }
        else rsym=false;																									//<vs_syymmetry>
      }
    }
    //-Sum results together. | Almacena resultados.
    if(shift||arp1||acep1.x||acep1.y||acep1.z||visc) {
      if(tdensity!=DDT_None) {
        if(delta)delta[p1]=(delta[p1]==FLT_MAX||deltap1==FLT_MAX ? FLT_MAX : delta[p1]+deltap1);
        else if(deltap1!=FLT_MAX)arp1+=deltap1;
      }
      ar[p1]+=arp1;
      ///////////////////////Still thinking of whether the artificial viscosity contribution to ace is needed for f-b Zhan's method
      if(Zhan_loop==0 || !boundp2) {
        ace[p1]=ace[p1]+acep1;
        }
      ////////////////////////
      const int th=omp_get_thread_num();
      if(visc>viscth[th*OMP_STRIDE])viscth[th*OMP_STRIDE]=visc;
      if(tvisco!=VISCO_Artificial) {
        gradvel[p1].a11+=gradvelp1.a11;
        gradvel[p1].a12+=gradvelp1.a12;
        gradvel[p1].a13+=gradvelp1.a13;
        gradvel[p1].a21+=gradvelp1.a21;
        gradvel[p1].a22+=gradvelp1.a22;
        gradvel[p1].a23+=gradvelp1.a23;
        gradvel[p1].a31+=gradvelp1.a31;
        gradvel[p1].a32+=gradvelp1.a32;
        gradvel[p1].a33+=gradvelp1.a33;
      }
      if(shift)shiftposfs[p1]=shiftposfsp1;
    }
  }
  //-Keep max value in viscdt. | Guarda en viscdt el valor maximo.
  for(int th=0; th<OmpThreads; th++)if(viscdt<viscth[th*OMP_STRIDE])viscdt=viscth[th*OMP_STRIDE];
//  Log->Printf("artificial viscosity");
//  Log->Printf("pos of particle 11163: %f %f %f",pos[11163].x,pos[11163].y,pos[11163].z);
//  Log->Printf("vel of particle 11163: %f %f %f",velrhop[11163].x,velrhop[11163].y,velrhop[11163].z);
//  Log->Printf("rho of particle 11163: %f",velrhop[11163].w);
//  Log->Printf("ar of particle 11163: %f",ar[11163]);
//  Log->Printf("ace of particle 11163 %f %f %f",ace[11163].x,ace[11163].y,ace[11163].z);
//  Log->Printf("gradvel of partice 11163 %f %f %f %f %f %f %f %f %f",gradvel[11163].a11,gradvel[11163].a12,gradvel[11163].a13,gradvel[11163].a21,gradvel[11163].a22,gradvel[11163].a23,gradvel[11163].a31,gradvel[11163].a32,gradvel[11163].a33);
}

////==============================================================================
///// Computes sub-particle stress tensor (Tau) for SPS turbulence model.   
//// The SPS model is disabled in the v5.0 NN version
////==============================================================================
//void JSphCpu::ComputeSpsTau(unsigned n, unsigned pini, const tfloat4 *velrhop, const tsymatrix3f *gradvel, tsymatrix3f *tau)const {
//  const int pfin = int(pini + n);
//#ifdef OMP_USE
//#pragma omp parallel for schedule (static)
//#endif
//  for (int p = int(pini); p<pfin; p++) {
//      const tsymatrix3f gradvel = SpsGradvelc[p];
//      const float pow1 = gradvel.xx*gradvel.xx + gradvel.yy*gradvel.yy + gradvel.zz*gradvel.zz;
//      const float prr = pow1 + pow1 + gradvel.xy*gradvel.xy + gradvel.xz*gradvel.xz + gradvel.yz*gradvel.yz;
//      const float visc_sps = SpsSmag*sqrt(prr);
//      const float div_u = gradvel.xx + gradvel.yy + gradvel.zz;
//      const float sps_k = (2.0f / 3.0f)*visc_sps*div_u;
//      const float sps_blin = SpsBlin*prr;
//      const float sumsps = -(sps_k + sps_blin);
//      const float twovisc_sps = (visc_sps + visc_sps);
//      const float one_rho2 = 1.0f / velrhop[p].w;
//      tau[p].xx = one_rho2*(twovisc_sps*gradvel.xx + sumsps);
//      tau[p].xy = one_rho2*(visc_sps   *gradvel.xy);
//      tau[p].xz = one_rho2*(visc_sps   *gradvel.xz);
//      tau[p].yy = one_rho2*(twovisc_sps*gradvel.yy + sumsps);
//      tau[p].yz = one_rho2*(visc_sps   *gradvel.yz);
//      tau[p].zz = one_rho2*(twovisc_sps*gradvel.zz + sumsps);
//  }
//}

//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid (forces and DEM) for NN using the SPH approach
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void JSphCpu::Interaction_ForcesCpuT_NN_SPH(const stinterparmsc &t,StInterResultc &res,int &Zhan_loop)const
{
  
  float viscdt=res.viscdt;
  float viscetadt=res.viscetadt;
  // time increment for hypoplastic or elastic model for verlet scheme,Symplectic scheme is pre-calculated in t.dt
  double dt = 0;
  if(t.npf) {
    if(Zhan_loop != 2){
      //Pressure gradient, velocity gradients (symetric)
      //hypoplasticity & elasticity: velocity gradient, artifacial viscosity, & density gradient
      //-Interaction Fluid-Fluid.
      InteractionForcesFluid_NN_SPH_PressGrad<tker,ftmode,tvisco,tdensity,shift>(t.npf,t.npb,false,Visco
        ,t.divdata,t.dcell,t.spsgradvel,t.pos,t.velrhop,t.code,t.idp,t.press
        ,viscdt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs,Zhan_loop);
      //-Interaction Fluid-Bound.
      if(Zhan_loop !=1){
      InteractionForcesFluid_NN_SPH_PressGrad<tker,ftmode,tvisco,tdensity,shift>(t.npf,t.npb,true,Visco*ViscoBoundFactor
        ,t.divdata,t.dcell,t.spsgradvel,t.pos,t.velrhop,t.code,t.idp,t.press
        ,viscdt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs,Zhan_loop);
      }
      if((tvisco!=VISCO_Artificial) & (tvisco!=VISCO_Hypoplasticity) & (tvisco!=VISCO_Elasticity) ) {
        //Build strain rate tensor and spin tensor and compute eta_visco //what will happen here for boundary particles?
        InteractionForcesFluid_NN_SPH_Visco_eta<ftmode,tvisco>(t.npf,t.npb,Visco
          ,t.visco_eta,t.velrhop,t.spsgradvel,t.d_tensor,t.w_tensor,t.auxnn,viscetadt,t.pos,t.code,t.idp);
      }
    }
      if ((tvisco != VISCO_ConstEq) & (tvisco!=VISCO_Hypoplasticity) & (tvisco!=VISCO_Elasticity)) { //artificial viscosity os included in PressGrad for hypo model
          //Morris - artificial viscosity 
          //-Interaction Fluid-Bound. 
          InteractionForcesFluid_NN_SPH_Morris<tker, ftmode, tvisco, tdensity, shift>(t.npf, t.npb, false, Visco
              , t.divdata, t.dcell, t.visco_eta, t.spstau, t.spsgradvel, t.auxnn, t.pos, t.velrhop, t.code, t.idp, t.ace);
          //-Interaction Fluid-Bound.     
          InteractionForcesFluid_NN_SPH_Morris<tker, ftmode, tvisco, tdensity, shift>(t.npf, t.npb, true, Visco*ViscoBoundFactor
              , t.divdata, t.dcell, t.visco_eta, t.spstau, t.spsgradvel, t.auxnn, t.pos, t.velrhop, t.code, t.idp, t.ace);
      }else{
          if (Time_inc==0){
            dt = double(CFLnumber * ((KernelH) / (max(Cs0, VelMax * 10.) + (KernelH)*viscdt)));
          }else{
            dt = Time_inc;
          }

          if(Zhan_loop != 2){
          // //-update stress & density rate in for boundary particles 
          // if (tvisco==VISCO_Hypoplasticity){
          //   InteractionForcesBound_NN_SPH<tker, ftmode, tvisco>(t.npb, 0, t.divdata, t.dcell
          //   , t.spstau, t.pos, t.velrhop, t.code, t.idp, viscdt, t.ar, dt);
          // }
          // hypoplastic and elastic model: Calculate strain and spin rate tensor, update stress, no need for boundary particles
          InteractionForcesFluid_NN_SPH_Visco_Stress_tensor<ftmode, tvisco>(t.npf, t.npb, Visco
              , t.visco_eta, t.void_ratio, t.spstau, t.spsgradvel,t.d_tensor, t.w_tensor, t.velrhop, t.auxnn, t.pos, t.code, t.idp, dt);

          //-Compute viscous terms and add them to the acceleration       
          InteractionForcesFluid_NN_SPH_ConsEq<tker, ftmode, tvisco, tdensity, shift>(t.npf, t.npb, false, Visco
              , t.divdata, t.dcell, t.visco_eta, t.spstau, t.auxnn, t.pos, t.velrhop, t.code, t.idp, t.ace, t.BoundNormalc, t.BoundCornerc,dt);
        }
          //-Interaction Fluid-Bound.
          if (Zhan_loop == 0){ // Not the Zhan's frictional method. f-b interaction.  
          InteractionForcesFluid_NN_SPH_ConsEq<tker, ftmode, tvisco, tdensity, shift>(t.npf, t.npb, true, Visco * ViscoBoundFactor
              , t.divdata, t.dcell, t.visco_eta, t.spstau, t.auxnn, t.pos, t.velrhop, t.code, t.idp, t.ace, t.BoundNormalc, t.BoundCornerc,dt);
          }
          if (Zhan_loop == 2){ // When Zhan_loop = 1: prediction step without boundary effect. Zhan_loop = 2: correction considering only the boundary effect.  
          InteractionForcesFluid_NN_SPH_ConsEq_Zhan_bound<tker, ftmode, tvisco, tdensity, shift>(t.npf, t.npb, true, Visco * ViscoBoundFactor
              , t.divdata, t.dcell, t.visco_eta, t.spstau, t.Force, t.auxnn, t.pos, t.velrhop, t.code, t.idp, t.ace, t.BoundNormalc, t.BoundCornerc,dt);
          } 
      }
    
    //-Interaction of DEM Floating-Bound & Floating-Floating. //(DEM)
    if (UseDEM)InteractionForcesDEM(CaseNfloat, t.divdata, t.dcell
        , FtRidp, DemData, t.pos, t.velrhop, t.code, t.idp, viscdt, t.ace);
  }
  if ((t.npbok) & (tvisco!=VISCO_Hypoplasticity) & (tvisco!=VISCO_Elasticity)) {
      //-Interaction Bound-Fluid.
      InteractionForcesBound_NN_SPH<tker, ftmode, tvisco>(t.npb, 0, t.divdata, t.dcell
          , t.spstau, t.pos, t.velrhop, t.code, t.idp, viscdt, t.ar, dt);
  }
  res.viscdt = viscdt;
  res.viscetadt = viscetadt;
}
//end_of_file