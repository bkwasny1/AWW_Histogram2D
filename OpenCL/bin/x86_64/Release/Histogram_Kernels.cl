__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

uint4 rgbToHsv(uint4 rgba)
{
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
    uint V = (uint)(maxC*255.f);       // [0,255]
    return (uint4)(H,S,V,255);
}

uint calculateCoordOfHist2D(uint4 HSV, const uint hBins, const uint sBins )
{
    uint h = HSV.x; // 0–179
    uint s = HSV.y; // 0–255
    uint hIdx = (uint)((float)h * (float)(hBins) / 180.0f);
    uint sIdx = (uint)((float)s * (float)(sBins) / 256.0f);
    if (hIdx >= hBins) hIdx = hBins - 1;
    if (sIdx >= sBins) sIdx = sBins - 1;

    uint idx = hIdx * sBins + sIdx;
    return idx;

}

__kernel void histogram2D(__read_only image2d_t inputImage, __write_only image2d_t outputImage,__global uint* outputHist , const uint width, const uint height, const uint hBin, const uint sBin)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    if(coord.x >= width || coord.y >= height)
    {
	    return;
    }
    uint4 rgba = read_imageui(inputImage, imageSampler, coord);

    uint4 HSV = rgbToHsv(rgba);
    uint idx = calculateCoordOfHist2D(HSV,hBin,sBin);

    write_imageui(outputImage, coord, HSV); //Zapis HSV jako wyjście

    atomic_inc(&outputHist[idx]);
}