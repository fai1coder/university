#pragma once
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <fstream>
#include <string>

//store misc openCL data
struct cl_enviroment {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	size_t max_local_size;
	bool set;
};

//printing error
inline void printErr(const char* errTxt);
//set cl_enviroment - platform, device, context
void setCL(cl_enviroment& env);
//compile openCL program
cl_program compilePr(const cl_enviroment& env, const char* filename);
