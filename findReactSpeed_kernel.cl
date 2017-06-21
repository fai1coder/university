

__kernel void findReactSpeed_kernel( __global float *f, __global float *i, __private int Nr, __private float h,__private float t,__private float k0,__private float k1, __private float b, __private int row_num, __private int total_num){
	int local_id = get_local_id(0);
	int id = Nr*(local_id/row_num) + local_id%row_num +  get_group_id(0)*row_num;

	if (id+Nr<total_num){
		float new_intensity = mad(-mad(k0 - k1, f[id], k1)*h, i[id] , i[id]);
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		i[id + Nr] = new_intensity;	
	}	

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	f[id] = -b*t*i[id] * f[id] + f[id];	
}		

