#version 460

layout (binding = 0) uniform sampler2D samplerColorMap;
layout (binding = 1) uniform sampler2D samplerGradientRamp;

layout (location = 0) in vec4 inColor;
layout (location = 1) in float inGradientPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inWorldPos;

layout (location = 0) out vec4 outFragColor;

// Simple lighting
void main() 
{
    // Calculate sphere normal from point coordinates
    vec2 coords = gl_PointCoord * 2.0 - 1.0;
    float radius = dot(coords, coords);
    if (radius > 1.0) {
        discard; // Outside of circle
    }
    
    // Calculate normal for lighting
    vec3 normal = normalize(vec3(coords, sqrt(1.0 - radius)));
    
    // Basic lighting
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.0));
    float diffuse = max(dot(normal, lightDir), 0.2);
    
    // Get color from gradient
    vec3 color = texture(samplerGradientRamp, vec2(inGradientPos, 0.0)).rgb;
    
    // Apply lighting
    outFragColor.rgb = color * diffuse;
    outFragColor.a = 1.0;
}
