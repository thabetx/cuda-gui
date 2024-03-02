#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdio.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

__global__ void offset_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int width, int height, unsigned char offset)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel;
	surf2Dread(&pixel, src, col * 4, row);
	pixel.x += offset;
	pixel.y += offset;
	surf2Dwrite(pixel, dst, col * 4, row);
}

__global__ void commit_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int width, int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel;
	surf2Dread(&pixel, src, col * 4, row);
	surf2Dwrite(pixel, dst, col * 4, row);
}
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
__global__ void transpose_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int width, int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel;
	surf2Dread(&pixel, src, col * 4, row);
	surf2Dwrite(pixel, dst, row * 4, col);
}

GLFWwindow* window;
unsigned int original_texture;
unsigned int texture;
cudaSurfaceObject_t original_surface;
cudaSurfaceObject_t surface;
const size_t width = 1024;
const size_t height = 1024;

void init()
{
	float* image = new float[width * height * 4];
	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			image[(width * row + col) * 4 + 0] = (row % 32) / 32.0f;
			image[(width * row + col) * 4 + 1] = (col % 32) / 32.0f;
			image[(width * row + col) * 4 + 2] = (float)row / width;
			image[(width * row + col) * 4 + 3] = 255;
		}
	}

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, image);

	glGenTextures(1, &original_texture);
	glBindTexture(GL_TEXTURE_2D, original_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, image);

	{
		cudaGraphicsResource* cuda_resource;
		cudaGraphicsGLRegisterImage(&cuda_resource, (GLuint)texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
		cudaGraphicsMapResources(1, &cuda_resource);

		cudaArray* cuda_array;
		cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);

		cudaResourceDesc res_desc{};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = cuda_array;

		cudaCreateSurfaceObject(&surface, &res_desc);
	}
	{
		cudaGraphicsResource* cuda_resource;
		cudaGraphicsGLRegisterImage(&cuda_resource, (GLuint)original_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
		cudaGraphicsMapResources(1, &cuda_resource);

		cudaArray* cuda_array;
		cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);

		cudaResourceDesc res_desc{};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = cuda_array;

		cudaCreateSurfaceObject(&original_surface, &res_desc);
	}
}

void frame()
{
	ImGui::Begin("Main Window", 0, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

	{
		ImGuiIO& io = ImGui::GetIO();

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

		const size_t samples_count = 1000;
		static float samples[samples_count];
		static size_t values_offset = 0;
		samples[values_offset] = io.Framerate;
		values_offset += 1;
		values_offset %= 1000;
		float average = 0.0f;
		for (int n = 0; n < samples_count; n++)
			average += samples[n];
		average /= (float)samples_count;
		char overlay[32];
		sprintf(overlay, "avg %f", average);
		ImGui::PlotLines("FPS", samples, samples_count, values_offset, overlay, 500, 900, ImVec2(0, 80.0f));
	}

	{
		static bool apply = true;
		static bool sync = true;
		ImGui::Checkbox("Apply Filter", &apply);
		ImGui::Checkbox("Synchronize", &sync);
		static int offset = 0;
		int min = 0, max = 255;
		ImGui::SliderScalar("Offset", ImGuiDataType_U8, &offset, &min, &max);
		if (apply)
		{
			dim3 blockSize(32, 32);
			dim3 gridSize(width / 32, height / 32);
			offset_kernel<<<gridSize, blockSize>>> (original_surface, surface, width, height, offset);
			if (sync)
				cudaDeviceSynchronize();
		}

		if (ImGui::Button("Tranpose"))
		{
			dim3 blockSize(32, 32);
			dim3 gridSize(width / 32, height / 32);
			transpose_kernel<<<gridSize, blockSize>>>(original_surface, surface, width, height);
			cudaDeviceSynchronize();
		}
		if (ImGui::Button("Commit"))
		{
			dim3 blockSize(32, 32);
			dim3 gridSize(width / 32, height / 32);
			commit_kernel<<<gridSize, blockSize>>>(surface, original_surface, width, height);
		}
		float image_height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY() - 30;
		ImGui::Image((void*)original_texture, { image_height, image_height });
		ImGui::SameLine();
		ImGui::Image((void*)texture, { image_height, image_height });
	}

	ImGui::End();

	static bool show_imgui_demo_window = false;
	if (ImGui::IsKeyPressed(ImGuiKey_F1))
		show_imgui_demo_window = !show_imgui_demo_window;
	if (show_imgui_demo_window)
		ImGui::ShowDemoWindow(&show_imgui_demo_window);
}

int main(int, char**)
{
	glfwInit();
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_MAXIMIZED, 1);
	window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	init();

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		int window_width, window_height;
		glfwGetWindowSize(window, &window_width, &window_height);
		ImGui::GetWindowWidth();
		ImGui::SetNextWindowPos({ 0, 0 });
		ImGui::SetNextWindowSize({ (float)window_width, (float)window_height });

		frame();

		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
