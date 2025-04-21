#version 460

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec4 inGradientPos;
layout (location = 2) in float inRadius;

layout (binding = 2) uniform UBO {
    mat4 projection;
    mat4 view;
    mat4 model;
} camera;

layout (location = 0) out vec4 outColor;
layout (location = 1) out float outGradientPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outWorldPos;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
};

void main() 
{
    // Calculate point size based on radius and distance to camera
    vec4 worldPos = camera.model * vec4(inPos, 1.0);
    vec4 viewPos = camera.view * worldPos;
    
    // Calculate normal for lighting (simplified for point sprite)
    outNormal = normalize(vec3(viewPos));
    outWorldPos = worldPos.xyz;
    
    // Output color and gradient position
    outColor = vec4(0.035);
    outGradientPos = inGradientPos.x;
    
    // Calculate position and point size
    gl_Position = camera.projection * viewPos;
    
    // Scale point size based on radius and distance
    float dist = length(viewPos.xyz);
    gl_PointSize = inRadius * 100.0 / dist;
}
