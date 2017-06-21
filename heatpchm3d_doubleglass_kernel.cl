

__kernel void heatpchm3d_doubleglass_kernel(__global float *in, __global float *out,  __global float *f, __global float *i, __private int Nr, __private float h, __private float a,__private float t,__private float C,__private float p,__private float k0,__private float k1, __private float b, __private float lambda, __private int row_num, __private int total_num, __private float k, __private float aG, __private float pG, __private float CG, __private float lambdaG,__private int idMin, __private int idMax) {
	int local_id = get_local_id(0);
	int id = Nr*(local_id/row_num) + local_id%row_num +  get_group_id(0)*row_num;
	int excess = id%Nr;

	if (id+Nr<total_num){
		float new_intensity;
		if (id > idMin && id < idMax) new_intensity = mad(-mad(k0 - k1, f[id], k1)*h, i[id] , i[id]);
		else new_intensity = i[id]*(-k*h+1);
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		i[id + Nr]  = new_intensity;
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
		if (excess!= 0 && excess != Nr-1 && id>=Nr) {
			float r = (excess)*h;
			float Q = -(i[id + Nr] - i[id]) / h;
			if (id > idMin && id < idMax) {out[id] = in[id] + mad(a*t, ((in[id + 1] + in[id - 1] + in[id + Nr] + in[id - Nr] - 4 * in[id]) / (h*h) + (1 / r) * (in[id + 1] - in[id - 1]) / (2 * h)), Q*t / (C*p));}
			else out[id] = in[id] + mad(aG*t, ((in[id + 1] + in[id - 1] + in[id + Nr] + in[id - Nr] - 4 * in[id]) / (h*h) + (1 / r) * (in[id + 1] - in[id - 1]) / (2 * h)), Q*t / (CG*pG));
		}
	}
	if (id>idMin && id < idMax) f[id] = -b*t*i[id] * f[id] + f[id];

	if (local_id < row_num) out[id] = lambdaG*in[id + Nr] / (lambdaG - 10.0*h);
	if (local_id >= row_num*((total_num/Nr)-1)) out[id] = lambdaG*in[id - Nr] / (lambdaG - 10.0*h);

	if (excess == 0) out[id] = out[id + 1];


}