#include <iostream>
#include <string>
#include "dnn.hpp"

using namespace std;

//Define parameters for the two convolution cases
#define Ni 64
#define Nn 64

#define Kx 3
#define Ky 3

#define Nx 224
#define Ny 224


//Define the parameters if not defined externally
#ifndef Sy
#define Sy 1
#define Sx 1
#endif

#ifndef Tnn
//Tiling Sizes
#define Tnn 32
#define Tn  16
#define Ti  16

#define Ty  8
#define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];
VTYPE (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];


VTYPE (*cuda_result)[NYSCL][NXSCL][Nn]; // memory to hold cuda result

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {      
      for(int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

std::pair<int,int> convolution_layer_blocked(
                                  VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                                  VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                                  VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]){
  //int c1=0,c2=0;
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]){
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

//cuda convolution function
__global__ void conv_cu(    VTYPE(&synapse)[Nn][Ni][Ky][Kx],
                            VTYPE(&neuron_i)[Ni][NYPAD][NXPAD],
                            VTYPE(&neuron_n)[Nn][NYSCL][NXSCL]
                        )
{
    if(blockIdx.x*1024+threadIdx.x < (Ny*Nx))
    {
        int ix = ((blockIdx.x * 1024) + threadIdx.x) % Nx;
        int iy = ((blockIdx.x * 1024) + threadIdx.x) / Nx;
        
        for(int out=0;out<Nn;out++)
        {
            VTYPE sum = 0;
            for(int y=iy;y<iy+3;y++)
            {
                for(int x=ix;x<ix+3;x++)
                {
                    for(int in=0;in<Ni;in++)
                    {
                        sum += neuron_i[in][y][x] * synapse[out][in][y-iy][x-ix];
                    }
                }
            }


	
            if(sum < 0)
            {
                neuron_n[out][iy][ix] = sum/4;
            }
            else
            {
                neuron_n[out][iy][ix] = sum;
            }

        }
    }
    
}


int main(const int argc, const char** argv) {
  
    cout << "allocating memory\n";
    synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])aligned_malloc(64,SYNAPSE_SIZE*sizeof(VTYPE));
    neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
    neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
    neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));

    //declare memory for cuda arrays
    VTYPE(*synapse_cu)[Nn][Ni][Ky][Kx];
    VTYPE(*neuron_i_cu)[Ni][NYPAD][NXPAD];
    VTYPE(*neuron_n_cu)[Nn][NYSCL][NXSCL];
    
    //memory for result since cuda array dimensions are different
    
    cuda_result = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
    
    //allocate memory for cuda arrays
    cudaMallocManaged(&synapse_cu,SYNAPSE_SIZE*sizeof(VTYPE));
    cudaMallocManaged(&neuron_i_cu,NYPAD*NXPAD*Ni*sizeof(VTYPE));
    cudaMallocManaged(&neuron_n_cu,NYSCL*NXSCL*Nn*sizeof(VTYPE));
    
    cout << "initializing arrays\n";
    fill_convolution_shared_simple(*synapse,*neuron_i);

    //copy to memory allocated for cuda
    //must do manually since dimensions are different
    
    //neuron input
    for(int i=0;i<NYPAD;i++)
    {
        for(int j=0;j<NXPAD;j++)
        {
            for(int in=0;in<Ni;in++)
            {
                (*neuron_i_cu)[in][i][j] = (*neuron_i)[i][j][in];
            }
        }
    }
    cout << "neuron input copied\n";
    
    //synapse
    for(int i=0;i<Ky;i++)
    {
        for(int j=0;j<Kx;j++)
        {
            for(int in=0;in<Ni;in++)
            {
                for(int out=0;out<Nn;out++)
                {
                    (*synapse_cu)[in][out][i][j] = (*synapse)[i][j][in][out];
                }
            }
        }
    }
    cout << "synapse copied\n";
    
    cout << "starting computation\n";
    //Simple Version
    begin_roi();
    convolution_layer(*synapse,*neuron_i,*neuron_n);
    end_roi();
    cout << "simple version complete!\n";

    //Blocked Version
    begin_roi();
    convolution_layer_blocked(*synapse,*neuron_i,*neuron_n2);
    end_roi();
    cout << "blocked computation complete!\n";

    //cuda Version
    begin_roi();
    int N = 1<<20; //HOW MANY ELEMENTS??
    int blockSize = 1024;
    int numBlocks = (N + blockSize-1) / blockSize;
    conv_cu <<<numBlocks, blockSize>>> (*synapse_cu, *neuron_i_cu, *neuron_n_cu);
    cudaDeviceSynchronize();
    end_roi();
    
    //store result into cuda_result
    cout << "storing cuda result\n";
    for(int i=0;i<NYSCL;i++)
    {
        for(int j=0;j<NXSCL;j++)
        {
            for(int out=0;out<Nn;out++)
            {
                (*cuda_result)[i][j][out] = (*neuron_n_cu)[out][i][j];
            }
        }
    }
    cout << "cuda complete!\n";
    
    //compare results
    compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);
    compare((VTYPE*)*neuron_n,(VTYPE*)*cuda_result,NYSCL*NXSCL*Nn);
    
    //free memory allocated for cuda
    cudaFree(synapse_cu);
    cudaFree(neuron_i_cu);
    cudaFree(neuron_n_cu);
    
    cout << "done\n";
}



