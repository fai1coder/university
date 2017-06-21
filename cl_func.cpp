#include "cl_func_header.h"

inline void printErr(const char* errTxt) {
	std::cout << errTxt << std::endl;
}

void setCL(cl_enviroment& env) {
	cl_platform_id *platforms;
	cl_uint plat_num;
	int err;
	size_t work_gr_size;

	// access platforms
	err = clGetPlatformIDs(0, NULL, &plat_num);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * plat_num);
	err = clGetPlatformIDs(plat_num, platforms, NULL);
	if (err < 0) { printErr("Couldn't identify a platform"); env.set = false; return; }
	//find gpu
	for (cl_uint i = 0; i<plat_num; ++i) {
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &env.device, NULL);
		if (!err) break;
	}
	if (err < 0) { printErr("Couldn't access devices");  env.set = false; return; }
	err = clGetDeviceInfo(env.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void*)&work_gr_size, NULL);
	if (err < 0) { printErr("Couldn't access device params"); env.set = false; return; }
	//set context
	env.context = clCreateContext(NULL, 1, &env.device, NULL, NULL, &err);
	if (err < 0) { printErr("Couldn't create a context");  env.set = false; return; }

	//create queue
	env.queue = clCreateCommandQueue(env.context, env.device, 0, &err);
	if (err < 0) { printErr("Couldn't create a command queue"); env.set = false; return; }

	//get max work group size
	err = clGetDeviceInfo(env.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &env.max_local_size, NULL);
	if (err < 0) { printErr("Couldn't access device info"); env.set = false;  return; }
	env.set = true;	
}

cl_program compilePr(const cl_enviroment& env, const char* filename) {
	cl_program prog;
	std::FILE *prog_hndl;
	char *prog_buff;
	size_t prog_size, log_size;
	int err;

	// read program file and place content into buffer
	prog_hndl = fopen(filename, "rb");
	if (prog_hndl == NULL) printErr("Couldn't find the program file");
	fseek(prog_hndl, 0, SEEK_END);
	prog_size = ftell(prog_hndl);
	rewind(prog_hndl);
	prog_buff = (char*)malloc(prog_size + 1);
	prog_buff[prog_size] = '\0';
	fread(prog_buff, sizeof(char), prog_size, prog_hndl);
	fclose(prog_hndl);

	// read program from file
	prog = clCreateProgramWithSource(env.context, 1,(const char**)&prog_buff, &prog_size, &err);
	if (err < 0) { printErr("Couldn't create the program"); return NULL; }
	free(prog_buff);

	// build program
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {
		// find log's and print
		char *error_log;
		clGetProgramBuildInfo(prog, env.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		error_log = (char*)malloc(log_size + 1);
		error_log[log_size] = '\0';
		clGetProgramBuildInfo(prog, env.device, CL_PROGRAM_BUILD_LOG, log_size + 1, error_log, NULL);
		printErr(error_log);
		free(error_log);
		return NULL;
	}
	return prog;
}