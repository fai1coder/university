

__kernel void heatpchm3d_kernel(__global float *in, __global float *out,  __global float *f, __global float *i, __private int Nr, __private float h, __private float a,__private float t,__private float C,__private float p,__private float k0,__private float k1, __private float b, __private float lambda){
	int local_id = get_local_id(0);
	int id = Nr*(local_id/4) + local_id%4 +  get_group_id(0)*4;

	if (id+Nr<5400) {
		i[id + Nr] = (-(k1 + (k0 - k1)*f[id]))*h*i[id] + i[id];
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
		if (id%Nr != 0 && id%Nr != Nr-1 && id>=Nr) {
				float r = (id%Nr)*h;
				float Q = -(i[id + Nr] - i[id]) / h;
				out[id] = in[id] + a*t*((in[id + 1] + in[id - 1] + in[id + Nr] + in[id - Nr] - 4 * in[id]) / (h*h) + (1 / r) * (in[id + 1] - in[id - 1]) / (2 * h)) + Q*t / (C*p);
			}
	}
	f[id] = -b*t*i[id] * f[id] + f[id];
	if (local_id < 4) out[id] = lambda*in[id + Nr] / (lambda - 10.0*h);
	if (local_id > 211) out[id] = lambda*in[id - Nr] / (lambda- 10.0*h);

	if (id%Nr==0) out[id] = in[id + 1];
}