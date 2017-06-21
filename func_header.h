#pragma once
#include <math.h>
#include <vector>
#include "cl_func_header.h"

struct material {
	float b;
	float lambda;
	float C;
	float p;				//	density
	float k0, k1;	//	exponential transmittance coefficient of glass
	float a;
private: std::vector<float*> data;
		 inline void fill_data() {
			 data.push_back(&b);
			 data.push_back(&lambda);
			 data.push_back(&C);
			 data.push_back(&p);
			 data.push_back(&k0);
			 data.push_back(&k1);
			 data.push_back(&a);
		 }
public: material() : b(0), lambda(0), C(0), p(0), k0(0), k1(0), a(0) {
			fill_data();
		};
		material(int i) {
			fill_data();
			for (int i = 0; i < data.size(); ++i) *(data[i]) = i;

		}
		material(float b_, float lambda_, float C_, float p_, float k0_, float k1_) : b(b_), lambda(lambda_), C(C_), p(p_), k0(k0_), k1(k1_) {
			a = lambda / (C*p);
			fill_data();
		}
		bool is_correct() {
			return lambda && C && p;
		}
		float* operator[](int i) {
			if (i >= 0 && i < data.size()) return data[i];
		}

		void clear() {
			for (int i = 0; i < data.size(); ++i) *(data[i]) = 0;
		}

		material& operator=(const material& other) {
			if (this != &other) {
				this->b = other.b;
				this->lambda = other.lambda;
				this->C = other.C;
				this->p = other.p;
				this->k0 = other.k0;
				this->k1 = other.k1;
				this->a = other.a;
			}
			return *this;
		}
};

//perform all calculations
//the size of medium is the same as in the experiment
//@timeTotal - time to model
//@outTime - in this time moments results wil be written to output array
//@outTimeSize - size of @outTime
//@params - C, lambda, b
//@k - k0 and k1
//@calcTemp - if true, temperature and absorbtion are calculated, otherwise only absorbtion is
//@print - write results to files
//@id - files postfix 
float * heatingPCHM3D(int timeTotal, int * outTime, int outTimeSize, std::vector<float> params, std::pair<float, float> k, bool calcTemp, bool print, int id);
//@Lz - size (in metres) of medium in z-direction
float * heatingPCHM3D_dynamic(float Lz, int timeTotal, int * outTime, int outTimeSize, std::vector<float> params, std::pair<float, float> k, bool calcTemp, bool print, int id);

float * heatingPCHM3D_double_glass(float Lz, float LzGlass, material mainMat, material glass, int timeTotal, int * outTime, int outTimeSize, bool calcTemp, bool print, int id);

//printing data
//@out_z - file to write values in (0, z) for z = 0..Lz
//@out_r_z0 - file to write values in (r, 0) for r = 0..Lr
//@out_r_zh - file to write values in (r,Lz) for r = 0..Lr
//@arr - data array, temperature or F
//@Nr - number of mesh point in r-direction
//@Nz - number of point in z-direction
//@t - current time in sec
void sendData(std::ofstream &out_z, std::ofstream &out_r_z0, std::ofstream &out_r_zh, float *arr, int Nr, int Nz, float t);

void sendDataDoubled(std::ofstream &out_z, std::ofstream &out_r_z0, std::ofstream &out_r_zh, float *arr, int Nr, int Nz, float t, int idMin, int idMax);

