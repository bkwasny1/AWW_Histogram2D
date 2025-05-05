__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__kernel void rgb2hsv(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    uint4 rgba = read_imageui(inputImage, imageSampler, coord);

    float r = (float)rgba.x / 255.0f;
    float g = (float)rgba.y / 255.0f;
    float b = (float)rgba.z / 255.0f;

    float maxC = fmax(fmax(r, g), b);
    float minC = fmin(fmin(r, g), b);
    float delta = maxC - minC;

    float h = 0.0f;
    if (delta > 0.00001f)
    {
        if (maxC == r)
            h = 60.0f * fmod(((g - b) / delta), 6.0f);
        else if (maxC == g)
            h = 60.0f * (((b - r) / delta) + 2.0f);
        else
            h = 60.0f * (((r - g) / delta) + 4.0f);
    }

    if (h < 0.0f)
        h += 360.0f;

    float s = (maxC == 0.0f) ? 0.0f : (delta / maxC);

    // Skalowanie do podanych zakresów:
    uint H = (uint)(h / 2.0f);         // [0,180]
    uint S = (uint)(s * 255.0f);       // [0,255]
    float V = maxC;                    // [0,1]

    write_imageui(outputImage, coord, (uint4)(H, S, (uint)(V * 255.0f), 255));
}



__kernel void hsvHistogram(__read_only image2d_t hsvImage,
                           __global uint *histogram,
                           const int width,
                           const int height,
                           const int hBins,
                           const int sBins)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height)
        return;

    uint4 pixel = read_imageui(hsvImage, imageSampler, pos);
    uchar h = pixel.x; // 0–179
    uchar s = pixel.y; // 0–255

    int hIdx = (int)((float)h * (float)(hBins) / 180.0f);
    int sIdx = (int)((float)s * (float)(sBins) / 256.0f);
    if (hIdx >= hBins) hIdx = hBins - 1;
    if (sIdx >= sBins) sIdx = sBins - 1;

    int idx = hIdx * sBins + sIdx;

    atomic_inc(&histogram[idx]);
}