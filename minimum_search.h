#pragma once
#include <utility>
#include <vector>

//results data
struct result {
	int size;	// size of @tmp and @trans
	float * temp;	// temperature
	float * trans;	// transmittance
};
// returns difference between ideal and current values (of temperature if @compateTemp, of transmittance otherwise
float difference(result ideal, result curr, bool compareTemp);
// returns true if current result's line lay below ideal's 
bool isLessThanIdeal(result ideal, result current, bool compareTemp);
// searcjes for optimal value of parametr @params[@num]
// @timeTotal - total time in sec to model
// @outTime - array of time moments where @ideal.trans and @ideal.temp was measured
// @k - k0 and k1 - transmittance in the beginning and end
// @params - vector of changable parametres
// @deltas - vector of @params deltas - values to add or abstract from @params during search
// @ideal - experimental result
// @directlyProportional - true if parametr is directly proportional to function
void searchOptimalParam(int num, int timeTotal, int * outTime, std::pair<float, float> k, std::vector<float> &params, std::vector<float> deltas, result ideal, bool directlyProportional);