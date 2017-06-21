#include "heat_model.h"

void heat_model::print_data(std::ofstream & out_z, std::ofstream & out_r_z0, std::ofstream & out_r_zh, float * arr, int t, int id_min, int id_max)
{
	out_z << "t=" << t << ";";
	out_r_z0 << "t=" << t << ";";
	out_r_zh << "t=" << t << ";";
	for (int i = 0; i < size; i += Nr) {
		out_z << arr[i] << ";";
	}
	for (int i = id_min; i < id_min + Nr; ++i) {
		out_r_z0 << arr[i] << ";";
	}
	if (!id_max) {
		for (int i = Nr*(Nz - 1); i < size; ++i) {
			out_r_zh << arr[i] << ";";
		}
	}
	else {
		for (int i = id_max; i < id_max + Nr; ++i) {
			out_r_zh << arr[i] << ";";
		}
	}
	out_z << std::endl;
	out_r_z0 << std::endl;
	out_r_zh << std::endl;
}
float * heat_model::run_default(bool print, std::string file_ending) {
	if (!is_correct()) {
		std::cout << "model is incorrect\n";
		return nullptr;
	}
	if (!env.set) {
		std::cout << "Something went wrong with OpenCL\n";
		return nullptr;
	}
	
	cl_program prog;
	cl_kernel krnl1, krnl2;	// @krnl1 and @krnl2 for temperature caclulations (swap T and T_next), @krnl_trans for transmittance calculation
	int err;							// store opencl error code
	cl_mem t_in, t_out, f_mem, i_mem;	// T data to send to kernel

	std::ofstream trz0, trzh, tz, frz0, frzh, fz;

	if (print) {
		//output files
		std::string file_ending_csv = "_" + file_ending + ".csv";
		trz0.open("trz0" + file_ending_csv);
		trzh.open("trzh" + file_ending_csv);
		tz.open("tz" + file_ending_csv);
		frz0.open("frz0" + file_ending_csv);
		frzh.open("frzh" + file_ending_csv);
		fz.open("fz" + file_ending_csv);

		if (!trz0.is_open() || !trzh.is_open() || !tz.is_open() || !frz0.is_open() || !frzh.is_open() || !fz.is_open()) {
			std::cout << "unable to open files for printing\n";
			return nullptr;
		}
	}
	

	//setCL(env);
	
	if (Nz > env.max_local_size) {
		std::cout << "can't syncronize this\n";
		return nullptr;
	}
	
	int id_min = 0, id_max = 0;
	if (Lz_glass > 0.0f)  id_min = (Lz_glass / h)*Nr, id_max = ((Lz + Lz_glass) / h)*Nr;
	int row_num = floor(env.max_local_size / Nz);
	const size_t local_gr_size = row_num * Nz;
	const size_t global_gr_size = size;

	float *T, *T_next;	//	temperature in current and next moment
	float *F, *I;		//	reaction coefficient in current and next moment, inetnsity of laser beam
	T = new float[size];
	T_next = new float[size];
	F = new float[size];
	I = new float[size];

	int sec = 0;
	float * output;				// output array (temperature if @calcTemp, transmittance otherwise)
	output = new float[output_num];

	//model with protetion glass
	if (Lz_glass > 0.0f) {
		prog = compilePr(env, "heatpchm3d_doubleglass_kernel.cl");

		krnl1 = clCreateKernel(prog, "heatpchm3d_doubleglass_kernel", &err);
		if (err < 0) { printErr("Couldn't create a kernel");	return NULL; };

		krnl2 = clCreateKernel(prog, "heatpchm3d_doubleglass_kernel", &err);
		if (err < 0) { printErr("Couldn't create a kernel");	return NULL; };
	}
	//model only holo plate
	else {
		prog = compilePr(env, "heatpchm3d_dynamic_kernel.cl");

		krnl1 = clCreateKernel(prog, "heatpchm3d_dynamic_kernel", &err);
		if (err < 0) { printErr("Couldn't create a kernel");	return NULL; };

		krnl2 = clCreateKernel(prog, "heatpchm3d_dynamic_kernel", &err);
		if (err < 0) { printErr("Couldn't create a kernel");	return NULL; };
	}

	//initialize initial values of intensity, temperature and concetration
	for (int i = 0; i < size; i++) {
		float r = i*h;
		if (i < Nr) I[i] = I0*exp(-alpha* pow(r, 2));
		else I[i] = 0;
		T[i] = 0;
		T_next[i] = 0;
		F[i] = 1;
	}

	//create buffers
	t_in = clCreateBuffer(env.context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(float)*Nr*Nz, T, &err);
	if (err < 0) { printErr("Couldn't create a buffer object"); return NULL; }

	t_out = clCreateBuffer(env.context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(float)*Nr*Nz, T_next, &err);
	if (err < 0) { printErr("Couldn't create a buffer object"); return NULL; }

	f_mem = clCreateBuffer(env.context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(float)*Nr*Nz, F, &err);
	if (err < 0) { printErr("Couldn't create a buffer object"); return NULL; }

	i_mem = clCreateBuffer(env.context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(float)*Nr*Nz, I, &err);
	if (err < 0) { printErr("Couldn't create a buffer object"); return NULL; }

	//set kernel arguments
	err = clSetKernelArg(krnl1, 0, sizeof(cl_mem), &t_in);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 1, sizeof(cl_mem), &t_out);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 2, sizeof(cl_mem), &f_mem);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 3, sizeof(cl_mem), &i_mem);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 4, sizeof(int), &Nr);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 5, sizeof(float), &h);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 6, sizeof(float), &main_mat.a);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 7, sizeof(float), &t);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 8, sizeof(float), &main_mat.C);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 9, sizeof(float), &main_mat.p);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 10, sizeof(float), &main_mat.k0);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 11, sizeof(float), &main_mat.k1);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 12, sizeof(float), &main_mat.b);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 13, sizeof(float), &main_mat.lambda);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 14, sizeof(int), &row_num);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl1, 15, sizeof(int), &size);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	if (Lz_glass > 0.0f) {
		err = clSetKernelArg(krnl1, 16, sizeof(float), &protection_mat.k0);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl1, 17, sizeof(float), &protection_mat.a);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl1, 18, sizeof(float), &protection_mat.p);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl1, 19, sizeof(float), &protection_mat.C);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl1, 20, sizeof(float), &protection_mat.lambda);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl1, 21, sizeof(int), &id_min);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl1, 22, sizeof(int), &id_max);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };
	}

	err = clSetKernelArg(krnl2, 0, sizeof(cl_mem), &t_out);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 1, sizeof(cl_mem), &t_in);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 2, sizeof(cl_mem), &f_mem);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 3, sizeof(cl_mem), &i_mem);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 4, sizeof(int), &Nr);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 5, sizeof(float), &h);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 6, sizeof(float), &main_mat.a);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 7, sizeof(float), &t);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 8, sizeof(float), &main_mat.C);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 9, sizeof(float), &main_mat.p);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 10, sizeof(float), &main_mat.k0);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 11, sizeof(float), &main_mat.k1);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 12, sizeof(float), &main_mat.b);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 13, sizeof(float), &main_mat.lambda);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 14, sizeof(int), &row_num);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	err = clSetKernelArg(krnl2, 15, sizeof(int), &size);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

	if (Lz_glass > 0.0f) {
		err = clSetKernelArg(krnl2, 16, sizeof(float), &protection_mat.k0);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl2, 17, sizeof(float), &protection_mat.a);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl2, 18, sizeof(float), &protection_mat.p);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl2, 19, sizeof(float), &protection_mat.C);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl2, 20, sizeof(float), &protection_mat.lambda);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl2, 21, sizeof(int), &id_min);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };

		err = clSetKernelArg(krnl2, 22, sizeof(int), &id_max);
		if (err < 0) { printErr("Couldn't set a kernel argument"); return NULL; };
	}

	int output_time_index = 0;

	//main calculation loop
	for (int count = 0; t*count <= time_total + t; ++count) {
		// run kernel
		err = clEnqueueNDRangeKernel(env.queue, count % 2 ? krnl1 : krnl2, 1, NULL, &global_gr_size, &local_gr_size, 0, NULL, NULL);
		if (err < 0) { printErr("Couldn't enqueue the kernel"); return NULL; }
		clFinish(env.queue);

		//write data to file every second
		if (count*t >= sec) {
			std::cout << sec << std::endl;
			//read buffers
			if (print || sec == output_time[output_time_index]) {
				err = clEnqueueReadBuffer(env.queue, t_out, CL_TRUE, 0,
					sizeof(float)*Nr*Nz, T_next, 0, NULL, NULL);
				if (err < 0) { printErr("Couldn't read the output buffer"); return NULL; }

				err = clEnqueueReadBuffer(env.queue, f_mem, CL_TRUE, 0,
					sizeof(float)*Nr*Nz, F, 0, NULL, NULL);
				if (err < 0) { printErr("Couldn't read the output buffer"); return NULL; }
				clFinish(env.queue);
				//print temperature and transmittance
				if (print) {
					print_data(tz, trz0, trzh, T, count*t, id_min, id_max);
					print_data(fz, frz0, frzh, F, count*t, id_min, id_max);
				}
				// write to output array
				if (sec == output_time[output_time_index]) {
					output[output_time_index] = T[0];
					++output_time_index;
				}
			}	
			
			++sec;
		}
	}
	tz.close(); fz.close(); trz0.close(); trzh.close(); frz0.close(); frzh.close();
	//release context
	clReleaseMemObject(t_in);
	clReleaseMemObject(t_out);
	clReleaseMemObject(f_mem);
	clReleaseMemObject(i_mem);
	clReleaseKernel(krnl1);
	clReleaseKernel(krnl2);
	clReleaseProgram(prog);

	return output;
}
float * heat_model::run_bleaching_only(bool print, std::string file_ending) {
	if (!is_correct()) {
		std::cout << "model is incorrect\n";
		return nullptr;
	}
	if (!env.set) {
		std::cout << "Something went wrong with OpenCL\n";
		return nullptr;
	}
		if (Nz > env.max_local_size) {
		std::cout << "can't syncronize this\n";
		return nullptr;
	}
	cl_program prog;
	cl_kernel krnl_trans;	
	int err;				// store opencl error code
	cl_mem f_mem, i_mem;	// T data to send to kernel

	std::ofstream frz0, frzh, fz;

	if (print) {
		//output files
		std::string file_ending_csv = "_" + file_ending + ".csv";
		frz0.open("frz0" + file_ending_csv);
		frzh.open("frzh" + file_ending_csv);
		fz.open("fz" + file_ending_csv);

		if (!frz0.is_open() || !frzh.is_open() || !fz.is_open()) {
			std::cout << "unable to open files for printing\n";
			return nullptr;
		}
	}
	
	int id_min = 0, id_max = 0;
	if (Lz_glass > 0.0f)  id_min = (Lz_glass / h)*Nr, id_max = ((Lz + Lz_glass) / h)*Nr;
	int row_num = floor(env.max_local_size / Nz);
	const size_t local_gr_size = row_num * Nz;
	const size_t global_gr_size = size;

	float *F, *I;			//	transmittance and intensity
	F = new float[size];
	I = new float[size];

	int sec = 0;
	float * output;			// output array
	output = new float[output_num];

	prog = compilePr(env, "findReactSpeed_kernel.cl");
	krnl_trans = clCreateKernel(prog, "findReactSpeed_kernel", &err);
	if (err < 0) { printErr("Couldn't create a kernel");	return nullptr; };


	//initialize initial values of intensity, temperature and concetration
	for (int i = 0; i < size; i++) {
		float r = i*h;
		if (i < Nr) I[i] = I0*exp(-alpha * pow(r, 2));
		else I[i] = 0; 
		F[i] = 1;
	}

	//create buffers
	f_mem = clCreateBuffer(env.context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(float)*Nr*Nz, F, &err);
	if (err < 0) { printErr("Couldn't create a buffer object"); return nullptr; }

	i_mem = clCreateBuffer(env.context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, sizeof(float)*Nr*Nz, I, &err);
	if (err < 0) { printErr("Couldn't create a buffer object"); return nullptr; }

	//set kernel arguments
	err = clSetKernelArg(krnl_trans, 0, sizeof(cl_mem), &f_mem);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 1, sizeof(cl_mem), &i_mem);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 2, sizeof(int), &Nr);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 3, sizeof(float), &h);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 4, sizeof(float), &t);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 5, sizeof(float), &main_mat.k0);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 6, sizeof(float), &main_mat.k1);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 7, sizeof(float), &main_mat.b);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 8, sizeof(float), &row_num);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };

	err = clSetKernelArg(krnl_trans, 9, sizeof(float), &size);
	if (err < 0) { printErr("Couldn't set a kernel argument"); return nullptr; };
	
	int output_time_index = 0;

	//main calculation loop
	for (int count = 0; t*count <= time_total + t; ++count) {

		err = clEnqueueNDRangeKernel(env.queue, krnl_trans, 1, NULL, &global_gr_size, &local_gr_size, 0, NULL, NULL);
		if (err < 0) { printErr("Couldn't enqueue the kernel"); return nullptr; }
		clFinish(env.queue);

		//write data to file every second
		if (count*t >= sec) {
			std::cout << sec << "\n";
			if (print || sec == output_time[output_time_index]) {
				err = clEnqueueReadBuffer(env.queue, f_mem, CL_TRUE, 0,
					sizeof(float)*size, F, 0, NULL, NULL);
				if (err < 0) { printErr("Couldn't read the output buffer"); return nullptr; }
				clFinish(env.queue);
				if (print) {
				print_data(fz, frz0, frzh, F, count*t);
			}
			// write to output array
			if (sec == output_time[output_time_index]) {
				output[output_time_index] = exp(-(main_mat.k1 + (main_mat.k0 - main_mat.k1)*F[0])*Lz);
				++output_time_index;
			}
			}
			++sec;
		}
	}
	//close output files
	if (fz.is_open()) fz.close(); frz0.close(); frzh.close();
	//release context
	clReleaseMemObject(f_mem);
	clReleaseMemObject(i_mem);
	clReleaseKernel(krnl_trans);
	clReleaseProgram(prog);

	return output;
}
void heat_model::search_optimal_param(int param_num, float delta, float * ideal, float e, bool is_directly_proportional, int iter_num, bool main_material, bool compare_temp) {
	if (!is_correct()) {
		std::cout << "wrong parametres\n";
		return;
	}
	if (!env.set) {
		std::cout << "Something went wrong with OpenCL\n";
		return ;
	}
	material * m;
	if (main_material) m = &main_mat;
	else m = &protection_mat;
	float * curr;
	curr = new float[output_num];
	float diff = 0;					// current difference between ideal and modelled result					
	bool curr_is_less, best_is_less;//current result is below ideal, best result is below ideal
	float best_diff, best_param;	// best difference, best parametr's value
	std::vector<float> used_params;	// vector of already used params
	//search loop
	int n = 0;
	while (n < iter_num) {
		std::cout << "Iteration #" << n << "\n";
		std::cout << "Current param value: " << *(m->operator[](param_num)) << "\n";
		// compare temperature?
		if (compare_temp) curr = run_default(false, "");
		else curr = run_bleaching_only(false, "");
		std::cout << "Results:\n" ;
		for (int i = 0; i < output_num; ++i) std::cout << curr[i] << " ";
		std::cout << "\n";
		diff = difference(ideal, curr);					// difference between experiment and calculations
		curr_is_less = is_less_than_ideal(ideal, curr);	// calculations result lays beloew experimental
		std::cout << "Difference " << diff << "\n";
		if (diff <= e) return;
		else {
			used_params.push_back(*(m->operator[](param_num)));	// store parametres value in used
			float temp = *(m->operator[](param_num));
			//new best difference
			if (!n || diff < best_diff) {
				if (n && best_is_less != curr_is_less) {	// truth lays between previous and current best - one is above ideal and other is below
					while (std::find(used_params.begin(), used_params.end(), *(m->operator[](param_num))) != used_params.end()) {
						delta /= 2;
						*(m->operator[](param_num)) = (*(m->operator[](param_num)) + best_param) / 2;
					}
				}
				else {
					if ((curr_is_less && is_directly_proportional) || !curr_is_less && !is_directly_proportional) {
						*(m->operator[](param_num)) += delta;
						while (std::find(used_params.begin(), used_params.end(), *(m->operator[](param_num))) != used_params.end()) {
							delta /= 2;
							*(m->operator[](param_num)) = (*(m->operator[](param_num)) + temp) / 2;
						}
					}
					else {
						*(m->operator[](param_num)) -= delta;
						while (std::find(used_params.begin(), used_params.end(), *(m->operator[](param_num))) != used_params.end()) {
							delta /= 2;
							*(m->operator[](param_num)) = (*(m->operator[](param_num)) + temp) / 2;
						}
					}
				}
				best_diff = diff;
				best_param = temp;
				best_is_less = curr_is_less;
			}
			else {
				if (curr_is_less != best_is_less) {
					while (std::find(used_params.begin(), used_params.end(), *(m->operator[](param_num))) != used_params.end()) {
						delta /= 2;
						*(m->operator[](param_num)) = (*(m->operator[](param_num)) + best_param) / 2;
					}
				}
				else {
					*(m->operator[](param_num)) = best_param;
					if (temp < best_param) {
						*(m->operator[](param_num)) += delta;
						while (std::find(used_params.begin(), used_params.end(), *(m->operator[](param_num))) != used_params.end()) {
							*(m->operator[](param_num)) = best_param;
							delta /= 2;
							*(m->operator[](param_num)) += delta;
						}
					}
					else {
						*(m->operator[](param_num)) -= delta;
						while (std::find(used_params.begin(), used_params.end(), *(m->operator[](param_num))) != used_params.end()) {
							*(m->operator[](param_num)) = best_param;
							delta /= 2;
							*(m->operator[](param_num)) -= delta;
						}
					}
				}
			}
			++n;
		}

	}
	*(m->operator[](param_num)) = best_param;
}
float heat_model::difference(float* ideal, float* curr) {
	float compare_val = 0;
	for (int i = 0; i < output_num; ++i) compare_val += pow(ideal[i] - curr[i], 2);
	return sqrt(compare_val);
}
bool heat_model::is_less_than_ideal(float* ideal, float* curr) {
	float val = 0;
	for (int i = 0; i < output_num; ++i) val += ideal[i] - curr[i];
	return val >= 0;
}