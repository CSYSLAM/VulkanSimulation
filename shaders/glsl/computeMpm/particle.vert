#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inVel;
layout(location = 2) in float inMass;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
};

void main ()
{
    const float spriteSize = 0.005 * inMass;
    gl_Position = vec4(inPos.x, inPos.y, 0, 1);
    gl_PointSize = 2.0;
}
