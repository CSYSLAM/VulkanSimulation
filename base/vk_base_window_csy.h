#pragma once
#pragma comment(linker, "/subsystem:console")

#include <windows.h>
#include <glfw/glfw3.h>

struct GLFWwindow;
struct GLFWwindow* GetGLFWWindow();

void InitializeWindow(int width, int height, const char* name);
bool ShouldQuit();
void DestroyWindow();
