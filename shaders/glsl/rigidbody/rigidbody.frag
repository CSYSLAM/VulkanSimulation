#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec3 inViewVec;
layout (location = 3) in vec3 inLightVec;
layout (location = 4) in vec3 inPos;

layout (location = 0) out vec4 outFragColor;

void main() 
{
    // 计算点精灵上的法线（从中心到片段位置的归一化向量）
    vec2 coords = gl_PointCoord * 2.0 - 1.0;
    float radius = dot(coords, coords);
    
    // 丢弃超出圆形的片段
    if (radius > 1.0) {
        discard;
    }
    
    // 计算球体表面的法线
    vec3 N = normalize(vec3(coords, sqrt(1.0 - radius)));
    
    // 光照计算
    vec3 L = normalize(inLightVec);
    vec3 V = normalize(inViewVec);
    vec3 R = reflect(-L, N);
    
    // 环境光分量
    vec3 ambient = inColor * 0.2;
    
    // 漫反射分量
    vec3 diffuse = max(dot(N, L), 0.0) * inColor;
    
    // 镜面反射分量
    vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(0.5);
    
    // 组合光照分量
    vec3 color = ambient + diffuse + specular;
    
    // 输出最终颜色
    outFragColor = vec4(color, 1.0);
}
