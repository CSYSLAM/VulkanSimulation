#version 450

layout (location = 0) in vec4 inPosition;  // xyz = position, w = radius
layout (location = 1) in vec4 inVelocity;  // xyz = velocity, w = mass
layout (location = 2) in vec4 inColor;     // rgb = color, a = opacity

layout (binding = 0) uniform UBO 
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;
layout (location = 4) out vec3 outPos;

void main() 
{
    // 使用球的位置和半径
    vec3 position = inPosition.xyz;
    float radius = inPosition.w;
    
    // 传递颜色到片段着色器
    outColor = inColor.rgb;
    
    // 计算世界空间位置
    vec4 worldPos = ubo.model * vec4(position, 1.0);
    
    // 计算视图空间位置
    vec4 viewPos = ubo.view * worldPos;
    outPos = viewPos.xyz;
    
    // 视线向量（从点到相机）
    outViewVec = -viewPos.xyz;
    
    // 光源位置（简单的顶部光源）
    vec3 lightPos = vec3(0.0, 5.0, 0.0);
    vec4 lightViewPos = ubo.view * vec4(lightPos, 1.0);
    outLightVec = lightViewPos.xyz - viewPos.xyz;
    
    // 法线（对于点精灵，我们将在片段着色器中计算）
    outNormal = vec3(0.0, 0.0, 1.0);
    
    // 最终位置
    gl_Position = ubo.projection * viewPos;
    
    // 设置点大小（使用半径）
    gl_PointSize = radius * 100.0; // 放大点大小使其可见
}
