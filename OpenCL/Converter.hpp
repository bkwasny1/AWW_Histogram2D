#pragma once

class Converter
{
private:
    static const float parula[64][3];
public:
    static void valueToParulaColor(float value, unsigned char& r, unsigned char& g, unsigned char& b);
};