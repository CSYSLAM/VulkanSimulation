#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 vel;

layout(location = 0) out vec4 outColor;

// Keep this value in sync with the `maxSpeed` const in the compute shader.
const float maxSpeed = 10.0; 

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    gl_PointSize = 1.0;

    // Mix colors based on position and velocity.
    outColor = mix(
        0.2 * vec4(pos, abs(vel.x) + abs(vel.y), 1.0),
        vec4(1.0, 0.5, 0.8, 1.0),
        sqrt(length(vel) / maxSpeed)
    );
}