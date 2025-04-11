#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inVel;

layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
}ubo;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main ()
{
	vec4 eyePos = ubo.view * vec4(inPos.xyz, 1.0);
	gl_Position = ubo.projection * eyePos;
	gl_PointSize = 1.0;
	
	// 根据速度大小调整颜色
	float speed = length(inVel.xyz);
	float normalizedSpeed = min(speed / 0.5, 1.0); // 假设最大速度为0.5
	outColor = vec4(0.0, normalizedSpeed, 1.0 - normalizedSpeed, 0.5);
}