#pragma once
#include <vector>

struct material {
	float b;
	float lambda;
	float C;
	float p;		//	density
	float k0, k1;	//	exponential transmittance coefficient of glass
	float a;
private:
	std::vector<float*> data;
	inline void fill_data() {
		data.push_back(&b);
		data.push_back(&lambda);
		data.push_back(&C);
		data.push_back(&p);
		data.push_back(&k0);
		data.push_back(&k1);
		data.push_back(&a);
	}
public:
	material() : b(0), lambda(0), C(0), p(0), k0(0), k1(0), a(0) {
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