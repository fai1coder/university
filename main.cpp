#include <cstdlib>
#include <vector>
#include <iostream>
#include "heat_model.h"

int main()
{
	//model parametres
	int time_total = 100;
	int output_time[] = { 5,15,25,50,100 };	// on this time moments results will be written in array
	int output_time_size = 5;

	float h = 5e-5;
	float Lz = 1e-3, Lx = 5e-3;
	float R = 0.002;
	float I0 = 2500;

	//experiment results
	float * ideal_transmittance = new float[5]{ float(0.29), float(0.33), float(0.35), float(0.44), float(0.53) };
	float * ideal_temp = new float[5]{ float(2.0), float(3.3), float(3.8), float(4.5), float(5.4) };

	//main material - PMMA with PQ and quartz
	material mainMat, glass;
	mainMat.b = 1.66e-5, glass.b = 0;
	mainMat.lambda = 0.194, glass.lambda = 1.114;
	mainMat.C = 1247, glass.C = 858;
	mainMat.k0 = 528, mainMat.k1 = 214, glass.k0 = 36, glass.k1 = 36;
	mainMat.p = 1180, glass.p = 2510;
	mainMat.a = mainMat.lambda / (mainMat.C*mainMat.p);
	glass.a = glass.lambda / (glass.C*glass.p);

	heat_model hm(h, time_total, output_time_size, output_time, Lz, Lx, R, I0, mainMat);

	std::string  name = std::to_string((int)(Lz * 1000)) + " " + std::to_string((int)I0);
	hm.run_default(true, "2500");

	system("pause");

}
