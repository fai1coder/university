#pragma once
#include "cl_func_header.h"
#include "material.h"
class heat_model {
private:
	int Nr, Nz;
	int size, time_total, output_num;
	int * output_time;
	float h, t, Lz, Lz_glass, Lx, R, alpha;
	float I0;
	
	cl_enviroment env;
	material main_mat, protection_mat;

	void print_data(std::ofstream & out_z, std::ofstream & out_r_z0, std::ofstream & out_r_zh, float * arr, int t, int id_min = 0, int id_max = 0);
	float difference(float* ideal, float* curr);
	bool is_less_than_ideal(float* ideal, float* curr);
	bool is_correct() {
		return Nz > 3 && Nz > 3 && h > 0.0f && Lz >=3*h && time_total > 0 && R > 0.0f && I0 > 0.0f && main_mat.is_correct() && (Lz_glass == 0.0f || (protection_mat.is_correct() && Lz_glass >=3*h));
	}

public:

	heat_model(float h, int time_total, int output_num, int * output_time, float Lz, float Lx, float R, float I0, material main_mat, material protection_mat = 0, float Lz_glass = 0){
		if (!set_size(Lz, Lx, h, Lz_glass)) { this->Lz = 0; this->Lx = 0; this->h = 0; this->Lz_glass = 0; }
		if (!set_R(R)) { this->R = 0; alpha = 0; }
		if (!set_total_time(time_total)) this->time_total = 0;
		if (!set_intensity(I0)) this->I0 = 0;
		if (!set_output_time(output_time, output_num)) {
			this->output_num = 0;
			this->output_time = nullptr;
		}
		if (!set_main_mat(main_mat)) this->main_mat.clear();		
		if (!set_protection_mat(protection_mat)) this->protection_mat.clear();
		if (main_mat.a) t = 0.001*pow(h, 2) / main_mat.a; else t = 0;
		setCL(env);
	}
	heat_model(float h, int time_total, int output_num, float * output, float Lz, float Lx, float R, float I0, material main_mat) {

	}
	heat_model() :
		Nr(0), Nz(0), time_total(0), output_num(0), output_time(nullptr), h(0), Lz(0), Lz_glass(0), Lx(0), R(0), I0(0), main_mat(), protection_mat(), size(0), alpha(0), t(0) {

	}
	~heat_model() {
		clReleaseCommandQueue(env.queue);
		clReleaseContext(env.context);
	}
	inline bool set_size(float Lz, float Lx, float h, float Lz_glass = 0) {
		if (Lz >= 3*h && Lx >= 3*h && (Lz_glass == 0 || Lz_glass >=3*h ) ) {
			this->h = h; this->Lz = Lz;	this->Lx = Lx;
			Nr = Lx / h;
			if (Lz_glass == 0) {
				Nz = Lz / h;
				size = Nr*Nz;
				return true;
			}
			this->Lz_glass = Lz_glass;
			Nz = (Lz + 2 * Lz_glass) / h;
			size = Nr*Nz;
			return true;
		}
		return false;
	}
	inline bool set_R(float R) {
		if (R > 0) {
			this->R = R;
			alpha = 1 / pow(R, 2);
			return true;
		}
		return false;
	}
	inline bool set_main_mat(material main_mat) {
		if (main_mat.is_correct()) {
			this->main_mat = main_mat;
			t = 0.001*pow(h, 2) / main_mat.a;
			return true;
		}
		return false;
	}
	inline bool set_protection_mat(material protection_mat) {
		if (protection_mat.is_correct()) {
			this->protection_mat = protection_mat;
			return true;
		}
		return false;
	}
	inline bool set_total_time(int time_total) {
		if (time_total > t) {
			this->time_total = time_total;
			return true;
		}
		return false;
	}
	inline bool set_output_time(int * output, int output_num) {
		if (output && output_num > 0) {
			this->output_time = output;
			this->output_num = output_num;
			return true;
		}
		return false;
	}
	inline bool set_intensity(float I0) {
		if (I0 > 0) {
			this->I0 = I0;
			return true;
		}
		return false;
	}

	inline float get_Lz() {
		return Lz;
	}
	inline float get_Lz_glass() {
		return Lz_glass;
	}
	inline float get_R() {
		return R;
	}
	inline float get_h() {
		return h;
	}
	inline material get_main_mat() {
		return main_mat;
	}
	inline material get_protection_mat() {
		return protection_mat;
	}
	inline float get_intentisy() {
		return I0;
	}

	float * run_default(bool print, std::string file_ending = 0);
	float * run_bleaching_only(bool print, std::string file_ending);
	void search_optimal_param(int param_num, float delta, float * ideal, float e, bool is_directly_proportional, int iter_num, bool main_material, bool compare_temp);
};