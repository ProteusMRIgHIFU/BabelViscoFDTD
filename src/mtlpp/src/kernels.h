kernel void sqr(
    const device float *vIn [[ buffer(0) ]],
    device float *vOut [[ buffer(1) ]],
    device float *vOut50 [[ buffer(2) ]],
    uint id[[ thread_position_in_grid ]])
{
    vOut[id] = vIn[id] * vIn[id] + 1000;
    vOut50[id] = vOut[id] - 333;
}

kernel void sqr2(
    const device float *vIn [[ buffer(0) ]],
    device float *vOut [[ buffer(1) ]],
    device float *vOut50 [[ buffer(2) ]],
    uint id[[ thread_position_in_grid ]])
{
    vOut[id] = vIn[id] * vIn[id]*2 + 1000;
    vOut50[id] = vOut[id] - 333;
}
