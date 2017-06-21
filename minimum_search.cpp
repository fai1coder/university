#include "minimum_search.h"
#include "func_header.h"
#include <algorithm>

float difference(result ideal, result current, bool compareTemp) {
	float compare_val = 0;
	for (int i = 0; i < ideal.size; ++i) {
		if (compareTemp) compare_val += pow(ideal.temp[i] - current.temp[i], 2);
		else compare_val += pow(ideal.trans[i] - current.trans[i], 2);
	}
	return sqrt(compare_val);
}

bool isLessThanIdeal(result ideal, result current, bool compareTemp) {
	float val = 0;
	for (int i = 0; i < ideal.size; ++i) {
		if (compareTemp) val += ideal.temp[i] - current.temp[i];
		else val += ideal.trans[i] - current.trans[i];
	}
	return val >= 0;
}


void searchOptimalParam(int num, int timeTotal, int * outTime, std::pair<float, float> k, std::vector<float>& params, std::vector<float> deltas, result ideal, bool directlyProportional) {
	if (num) {
		std::cout << "Searchin for minimum of heating function " << std::endl;
		std::cout << "Param num: " << num << std::endl;
		std::cout << "Ideal temperature: " << std::endl;
		for (int i = 0; i < ideal.size; ++i) std::cout << ideal.temp[i] << " ";
	}
	else {
		std::cout << "Searchin for minimum of F function " << std::endl;
		std::cout << "Ideal transmittance: " << std::endl;
		for (int i = 0; i < ideal.size; ++i) std::cout << ideal.trans[i] << " ";
	}
	std::cout << std::endl;
	result current;
	current.size = ideal.size;

	float diff = 0;	// current difference between ideal and modelled result
	//current result is below ideal, best result is below ideal
	bool currResIsLess, bestResIsLess;
	float bestDiff, bestParam;	// best difference, best parametr's value
	std::vector<float> usedParams;	// vector of already used params

	float e; //precision
	if (num) e = ideal.temp[ideal.size - 1] * 0.1;
	else e = ideal.trans[ideal.size - 1] * 0.05;
	//search loop
	int n = 0;
	while (n < 10) {
		std::cout << "Iteration #" << n << std::endl;
		std::cout << "Current param value: " << params[num] << std::endl;
		//param[0] is speed so we can usetruncated kernel and do not calculate temperature
		if (num) current.temp = heatingPCHM3D(timeTotal, outTime, ideal.size, params, k, true, false, 0);
		else current.trans = heatingPCHM3D(timeTotal, outTime, ideal.size, params, k, false, false, 0);
		std::cout << "Results:" << std::endl;
		if (num) for (int i = 0; i < current.size; ++i) std::cout << current.temp[i] << " ";
		else for (int i = 0; i < current.size; ++i) std::cout << current.trans[i] << " ";
		std::cout << std::endl;
		diff = difference(ideal, current, num);
		currResIsLess = isLessThanIdeal(ideal, current, num);
		std::cout << "Difference " << diff << std::endl;
		if (diff <= e) return;
		else {
			usedParams.push_back(params[num]);
			float temp = params[num];
			//new best difference
			if (!n || diff < bestDiff) {
				if (n && bestResIsLess != currResIsLess) {	// truth lays between previous and current best - one is above ideal and other is below
					while (std::find(usedParams.begin(), usedParams.end(), params[num]) != usedParams.end()) {
						deltas[num] /= 2;
						params[num] = (params[num] + bestParam) / 2;
					}
				}
				else {	
					if ((currResIsLess && directlyProportional) || !currResIsLess && !directlyProportional) {	
							params[num] += deltas[num];
							while (std::find(usedParams.begin(), usedParams.end(), params[num]) != usedParams.end()) {
								deltas[num] /= 2;
								params[num] = (params[num] + temp) / 2;
						}
					}
					else {		
						params[num] -= deltas[num];
						while (std::find(usedParams.begin(), usedParams.end(), params[num]) != usedParams.end()) {
							deltas[num] /= 2;
							params[num] = (params[num] + temp) / 2;
						}
					}
				}
				bestDiff = diff;
				bestParam = temp;
				bestResIsLess = currResIsLess;
			}
			else {
				if (currResIsLess != bestResIsLess) {
					while (std::find(usedParams.begin(), usedParams.end(), params[num]) != usedParams.end()) {
						deltas[num] /= 2;
						params[num] = (params[num] + bestParam) / 2;
					}
				}
				else {
					params[num] = bestParam;
					if (temp < bestParam) {
						params[num] += deltas[num];
						while (std::find(usedParams.begin(), usedParams.end(), params[num]) != usedParams.end()) {
							params[num] = bestParam;
							deltas[num] /= 2;
							params[num] += deltas[num];
						}
					}
					else {
						params[num] -= deltas[num];
						while (std::find(usedParams.begin(), usedParams.end(), params[num]) != usedParams.end()) {
							params[num] = bestParam;
							deltas[num] /= 2;
							params[num] -= deltas[num];
						}
					}
				}
			}
			++n;
		}

	}
	params[num] = bestParam;
}