__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__constant float parula[64][3] = {
    {0.2081, 0.1663, 0.5292}, {0.2116, 0.1898, 0.5777}, {0.2123, 0.2138, 0.6270}, {0.2081, 0.2386, 0.6771},
    {0.1959, 0.2645, 0.7279}, {0.1707, 0.2919, 0.7792}, {0.1253, 0.3242, 0.8303}, {0.0591, 0.3598, 0.8683},
    {0.0117, 0.3875, 0.8820}, {0.0060, 0.4086, 0.8828}, {0.0165, 0.4266, 0.8786}, {0.0329, 0.4430, 0.8720},
    {0.0498, 0.4586, 0.8641}, {0.0629, 0.4737, 0.8554}, {0.0723, 0.4887, 0.8467}, {0.0779, 0.5040, 0.8384},
    {0.0793, 0.5200, 0.8312}, {0.0749, 0.5375, 0.8263}, {0.0641, 0.5570, 0.8240}, {0.0488, 0.5772, 0.8228},
    {0.0343, 0.5966, 0.8199}, {0.0265, 0.6137, 0.8135}, {0.0239, 0.6287, 0.8038}, {0.0231, 0.6418, 0.7913},
    {0.0228, 0.6535, 0.7768}, {0.0267, 0.6642, 0.7607}, {0.0384, 0.6743, 0.7436}, {0.0590, 0.6838, 0.7254},
    {0.0843, 0.6928, 0.7062}, {0.1133, 0.7015, 0.6859}, {0.1453, 0.7098, 0.6646}, {0.1801, 0.7177, 0.6424},
    {0.2178, 0.7250, 0.6193}, {0.2586, 0.7317, 0.5954}, {0.3022, 0.7376, 0.5712}, {0.3482, 0.7424, 0.5473},
    {0.3953, 0.7459, 0.5244}, {0.4420, 0.7481, 0.5033}, {0.4871, 0.7491, 0.4840}, {0.5300, 0.7491, 0.4661},
    {0.5709, 0.7485, 0.4494}, {0.6099, 0.7473, 0.4337}, {0.6473, 0.7456, 0.4188}, {0.6834, 0.7435, 0.4044},
    {0.7184, 0.7411, 0.3905}, {0.7525, 0.7384, 0.3768}, {0.7858, 0.7356, 0.3633}, {0.8185, 0.7327, 0.3498},
    {0.8507, 0.7299, 0.3360}, {0.8824, 0.7274, 0.3217}, {0.9139, 0.7258, 0.3063}, {0.9450, 0.7261, 0.2886},
    {0.9739, 0.7314, 0.2666}, {0.9938, 0.7455, 0.2403}, {0.9990, 0.7653, 0.2164}, {0.9955, 0.7861, 0.1967},
    {0.9880, 0.8066, 0.1794}, {0.9789, 0.8271, 0.1633}, {0.9697, 0.8481, 0.1475}, {0.9626, 0.8705, 0.1309},
    {0.9589, 0.8949, 0.1132}, {0.9598, 0.9218, 0.0948}, {0.9661, 0.9514, 0.0755}, {0.9763, 0.9831, 0.0538}
}; 

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

__kernel void histogram2DWithOutHsv(__read_only image2d_t inputImage, __global uint* outputHist , const uint width, const uint height, const uint hBin, const uint sBin)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    if(coord.x >= width || coord.y >= height)
    {
	    return;
    }
    uint4 rgba = read_imageui(inputImage, imageSampler, coord);

    uint4 HSV = rgbToHsv(rgba);
    uint idx = calculateCoordOfHist2D(HSV,hBin,sBin);

    atomic_inc(&outputHist[idx]);
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

__kernel void visualizeHistogram(__global const uint* histData, 
                               __global uchar* outputImage,
                               const uint width, 
                               const uint height,
                               const uint maxValue,
                               const uint histGray)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = (float)histData[idx] / (float)maxValue;
    value = clamp(value, 0.0f, 1.0f);
    
    if (histGray == 1)
    {
        int outIdx = (y * width + x) * 3;
        outputImage[outIdx + 0] = (uchar)(value * 255);
        outputImage[outIdx + 1] = (uchar)(value * 255);
        outputImage[outIdx + 2] = (uchar)(value * 255);
    }
    else
    {
        // Map to parula
        float scaled = value * 63.0f;
        int colorIdx = (int)scaled;
        float t = scaled - colorIdx;
        
        if (colorIdx >= 63) colorIdx = 62;
        
        float r = (1-t)*parula[colorIdx][0] + t*parula[colorIdx+1][0];
        float g = (1-t)*parula[colorIdx][1] + t*parula[colorIdx+1][1];
        float b = (1-t)*parula[colorIdx][2] + t*parula[colorIdx+1][2];
        
        int outIdx = (y * width + x) * 3;
        outputImage[outIdx + 0] = (uchar)(r * 255);
        outputImage[outIdx + 1] = (uchar)(g * 255);
        outputImage[outIdx + 2] = (uchar)(b * 255);
    }
}