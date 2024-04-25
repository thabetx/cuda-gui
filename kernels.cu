#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

__global__ void copy_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel;
	surf2Dread(&pixel, src, col * 4, row);
	surf2Dwrite(pixel, dst, col * 4, row);
}

__global__ void offset_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int offset)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel;
	surf2Dread(&pixel, src, col * 4, row);
	pixel.x += offset;
	pixel.y += offset;
	surf2Dwrite(pixel, dst, col * 4, row);
}

__global__ void transpose_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel;
	surf2Dread(&pixel, src, col * 4, row);
	surf2Dwrite(pixel, dst, row * 4, col);
}

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
__global__ void transpose_coalesced_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst)
{
	__shared__ uchar4 data[TILE_DIM + 1][TILE_DIM + 1];
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	uchar4 pixel;
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		surf2Dread(&pixel, src, x * 4, y+j);
		data[threadIdx.y + j][threadIdx.x] = pixel;
	}

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		surf2Dwrite(data[threadIdx.x][threadIdx.y + j], dst, x * 4, y+j);
	}
}

__global__ void clear_kernel(cudaSurfaceObject_t dst, uchar4 clear_color)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	surf2Dwrite(clear_color, dst, col * 4, row);
}

__global__ void blur_x_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int r, int width)
{
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	uchar4 dropped_pixel, entered_pixel;
	float scale = 1.0f / (r+r+1);

	int kernel_width = width / gridDim.x;
	int x = blockIdx.x*kernel_width;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	for (int i = -r-1; i < r; ++i)
	{
		int idx = i + x;
		if (idx < 0)           surf2Dread(&entered_pixel, src, 0 * 4, y);
		else if (idx >= width) surf2Dread(&entered_pixel, src, (width-1) * 4, y);
		else                   surf2Dread(&entered_pixel, src, idx * 4, y);

		sum.x += entered_pixel.x;
		sum.y += entered_pixel.y;
		sum.z += entered_pixel.z;
		sum.w += entered_pixel.w;
	}

	for (int i = 0; i < kernel_width; ++i)
	{
		int idx = x + i - r - 1;
		if (idx < 0) surf2Dread(&dropped_pixel, src, 0*4, y);
		else         surf2Dread(&dropped_pixel, src, idx*4, y);

		sum.x -= dropped_pixel.x;
		sum.y -= dropped_pixel.y;
		sum.z -= dropped_pixel.z;
		sum.w -= dropped_pixel.w;

		idx = x + i + r;
		if (idx >= width) surf2Dread(&entered_pixel, src, (width-1) * 4, y);
		else              surf2Dread(&entered_pixel, src, idx * 4, y);

		sum.x += entered_pixel.x;
		sum.y += entered_pixel.y;
		sum.z += entered_pixel.z;
		sum.w += entered_pixel.w;
		surf2Dwrite(make_uchar4(sum.x * scale, sum.y * scale, sum.z * scale, sum.w * scale), dst, (i + x) * 4, y);
	}
}

__global__ void blur_y_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int r, int width)
{
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	uchar4 dropped_pixel, entered_pixel;
	float scale = 1.0f / (r+r+1);

	int kernel_width = width / gridDim.x;
	int y = blockIdx.x*kernel_width;
	int x = blockIdx.y*blockDim.y + threadIdx.y;
	for (int i = -r-1; i < r; ++i)
	{
		int idx = i + y;
		if (idx < 0)           surf2Dread(&entered_pixel, src, x*4, 0);
		else if (idx >= width) surf2Dread(&entered_pixel, src, x*4, width-1);
		else                   surf2Dread(&entered_pixel, src, x*4, idx);

		sum.x += entered_pixel.x;
		sum.y += entered_pixel.y;
		sum.z += entered_pixel.z;
		sum.w += entered_pixel.w;
	}

	for (int i = 0; i < kernel_width; ++i)
	{
		int idx = y + i - r - 1;
		if (idx < 0) surf2Dread(&dropped_pixel, src, x*4, 0);
		else         surf2Dread(&dropped_pixel, src, x*4, idx);

		sum.x -= dropped_pixel.x;
		sum.y -= dropped_pixel.y;
		sum.z -= dropped_pixel.z;
		sum.w -= dropped_pixel.w;

		idx = y + i + r;
		if (idx >= width) surf2Dread(&entered_pixel, src, x*4, width-1);
		else              surf2Dread(&entered_pixel, src, x*4, idx);

		sum.x += entered_pixel.x;
		sum.y += entered_pixel.y;
		sum.z += entered_pixel.z;
		sum.w += entered_pixel.w;
		surf2Dwrite(make_uchar4(sum.x * scale, sum.y * scale, sum.z * scale, sum.w * scale), dst, x*4, y+i);
	}
}

GLFWwindow* window;
unsigned int original_texture;
unsigned int result_texture;
unsigned int temp_texture;
cudaSurfaceObject_t original_surface;
cudaSurfaceObject_t result_surface;
cudaSurfaceObject_t temp_surface;
int width;
int height;
cudaEvent_t start_event, end_event;
float elapsed_time_ms;
int kernel_rounds = 1;

void init()
{
	unsigned char* image = stbi_load("data\\h-ng-nguy-n-gnxqelrnKXs-unsplash.jpg", &width, &height, NULL, 4);

	if (image == nullptr)
	{
		width = 2048;
		height = 2048;
		stbi_image_free(image);
		image = new unsigned char[width * height * 4];
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t col = 0; col < width; ++col)
			{
				image[(width * row + col) * 4 + 0] = (row % 255);
				image[(width * row + col) * 4 + 1] = (col % 255);
				image[(width * row + col) * 4 + 2] = ((row+col) % 255);
				image[(width * row + col) * 4 + 3] = 255;
			}
		}
	}

	glGenTextures(1, &original_texture);
	glBindTexture(GL_TEXTURE_2D, original_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

	glGenTextures(1, &result_texture);
	glBindTexture(GL_TEXTURE_2D, result_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

	glGenTextures(1, &temp_texture);
	glBindTexture(GL_TEXTURE_2D, temp_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

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
	{
		cudaGraphicsResource* cuda_resource;
		cudaGraphicsGLRegisterImage(&cuda_resource, (GLuint)result_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
		cudaGraphicsMapResources(1, &cuda_resource);

		cudaArray* cuda_array;
		cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);

		cudaResourceDesc res_desc{};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = cuda_array;

		cudaCreateSurfaceObject(&result_surface, &res_desc);
	}
	{
		cudaGraphicsResource* cuda_resource;
		cudaGraphicsGLRegisterImage(&cuda_resource, (GLuint)temp_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
		cudaGraphicsMapResources(1, &cuda_resource);

		cudaArray* cuda_array;
		cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);

		cudaResourceDesc res_desc{};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = cuda_array;

		cudaCreateSurfaceObject(&temp_surface, &res_desc);
	}

	{
		cudaEventCreate(&start_event);
		cudaEventCreate(&end_event);
	}
}

void frame()
{
	static int offset = 0;
	static int blur_rounds = 1;
	static int kernel_radius = 5;
	static float clear_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

	ImGui::Begin("Main Window", 0, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

	static enum Kernel {
		none,
		copy,
		copy_back,
		shift,
		tranpose,
		tranpose_coalesced,
		blur_x,
		blur_y,
		blur_xy,
		clear,
		kernels_count
	} kernel = blur_x;

	{
		ImGui::BeginChild("Child Left", ImVec2(ImGui::GetContentRegionAvail().x * 0.3f, 0));

		ImGui::SameLine();
		{
			ImGui::BeginChild("Child Left 2", ImVec2(ImGui::GetContentRegionAvail().x * 0.3f, 300));

			ImGui::RadioButton("None", (int*)&kernel, none);
			ImGui::RadioButton("Copy >>>", (int*)&kernel, copy);
			ImGui::RadioButton("Copy <<<", (int*)&kernel, copy_back);
			ImGui::RadioButton("shift", (int*)&kernel, shift);
			ImGui::RadioButton("Transpose", (int*)&kernel, tranpose);
			ImGui::RadioButton("Transpose Coalesced", (int*)&kernel, tranpose_coalesced);
			ImGui::RadioButton("Blur X", (int*)&kernel, blur_x);
			ImGui::RadioButton("Blur Y", (int*)&kernel, blur_y);
			ImGui::RadioButton("Blur X+Y", (int*)&kernel, blur_xy);
			ImGui::RadioButton("Clear", (int*)&kernel, clear);

			ImGui::EndChild();
		}
		ImGui::SameLine();
		{
			ImGui::BeginChild("Child Left 3", ImVec2(0, 300));

			if (kernel == shift)
			{
				int min = 0, max = 255;
				ImGui::SliderScalar("Offset", ImGuiDataType_U8, &offset, &min, &max);
			}

			if (kernel == blur_x || kernel == blur_y || kernel == blur_xy)
			{
				ImGui::SliderInt("Kernel Radius", &kernel_radius, 1, 12);
			}

			if (kernel == blur_xy)
			{
				ImGui::SliderInt("Blur Rounds", &blur_rounds, 0, 20);
			}

			if (kernel == clear)
			{
				ImGui::ColorEdit4("Clear Color", clear_color);
			}
			ImGui::EndChild();
		}

		if (ImGui::IsKeyPressed(ImGuiKey_J) && (int)kernel < (int)(kernels_count-1)) kernel = Kernel(int(kernel) + 1);
		if (ImGui::IsKeyPressed(ImGuiKey_K) && (int)kernel > 0) kernel = Kernel(int(kernel) - 1);

		static Kernel previous_kernel;
		if (ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
			previous_kernel = kernel;
			kernel = copy;
		}
		if (ImGui::IsKeyReleased(ImGuiKey_Tab)) {
			kernel = previous_kernel;
		}
		if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
			previous_kernel = kernel;
			kernel = copy_back;
		}
		if (ImGui::IsKeyReleased(ImGuiKey_Enter)) {
			kernel = previous_kernel;
		}

		dim3 grid_size, block_size;
		uchar4 clear_color_u = make_uchar4(
			unsigned char(clear_color[0] * 255),
			unsigned char(clear_color[1] * 255),
			unsigned char(clear_color[2] * 255),
			unsigned char(clear_color[3] * 255));

		cudaEventRecord(start_event, 0);

		for (size_t kernel_round = 0; kernel_round < kernel_rounds; ++kernel_round)
		{
			switch (kernel)
			{
			case none:
				break;
			case copy:
			{
				block_size = dim3(32, 32);
				grid_size = dim3(width / 32, height / 32);
				copy_kernel << <grid_size, block_size >> > (original_surface, result_surface);
				break;
			}
			case copy_back:
			{
				block_size = dim3(32, 32);
				grid_size = dim3(width / 32, height / 32);
				copy_kernel << <grid_size, block_size >> > (result_surface, original_surface);
				break;
			}
			case shift:
			{
				block_size = dim3(32, 32);
				grid_size = dim3(width / 32, height / 32);
				offset_kernel << <grid_size, block_size >> > (original_surface, result_surface, offset);
				break;
			}
			case tranpose:
			{
				block_size = dim3(32, 32);
				grid_size = dim3(width / 32, height / 32);
				transpose_kernel << <grid_size, block_size >> > (original_surface, result_surface);
				break;
			}
			case tranpose_coalesced:
			{
				block_size = dim3(TILE_DIM, BLOCK_ROWS);
				grid_size = dim3(width / TILE_DIM, height / TILE_DIM);
				transpose_coalesced_kernel << <grid_size, block_size >> > (original_surface, result_surface);
				break;
			}
			case blur_x:
			{
				block_size = dim3(1, 16);
				grid_size = dim3(32, height/block_size.y);
				blur_x_kernel << <grid_size, block_size >> > (original_surface, result_surface, kernel_radius, width);
				break;
			}
			case blur_y:
			{
				block_size = dim3(1, 16);
				grid_size = dim3(32, height/block_size.y);
				blur_y_kernel << <grid_size, block_size >> > (original_surface, result_surface, kernel_radius, width);
				break;
			}
			case blur_xy:
			{
				block_size = dim3(32, 32);
				grid_size = dim3(width / 32, height / 32);
				copy_kernel << <grid_size, block_size >> > (original_surface, result_surface);

				for (size_t i = 0; i < blur_rounds; ++i)
				{
					block_size = dim3(1, 16);
					grid_size = dim3(32, height/block_size.y);
					blur_x_kernel << <grid_size, block_size >> > (result_surface, temp_surface, kernel_radius, width);

					block_size = dim3(1, 16);
					grid_size = dim3(32, height/block_size.y);
					blur_y_kernel << <grid_size, block_size >> > (temp_surface, result_surface, kernel_radius, width);
				}

				break;
			}
			case clear:
			{
				block_size = dim3(32, 32);
				grid_size = dim3(width / 32, height / 32);
				clear_kernel << <grid_size, block_size >> > (result_surface, clear_color_u);
				break;
			}
			default:
				break;
			}
		}

		cudaEventRecord(end_event, 0);
		cudaEventSynchronize(end_event);
		cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event);
		cudaDeviceSynchronize();

		ImGui::SliderInt("Kernel Samples", &kernel_rounds, 1, 30);
		ImGuiIO& io = ImGui::GetIO();
		ImGui::Text("%dx%d", width, height);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		ImGui::Text("Tab to show original. Enter to commit current image.");

		const size_t samples_count = 50;
		static float samples[samples_count];
		static size_t values_offset = 0;
		samples[values_offset] = elapsed_time_ms*1000/kernel_rounds;
		values_offset += 1;
		values_offset %= samples_count;
		float average = 0.0f;
		for (int n = 0; n < samples_count; n++)
			average += samples[n];
		average /= (float)samples_count;
		char overlay[32];
		sprintf(overlay, "avg %.0fus", average);
		ImGui::PlotLines("Kernel Time", samples, samples_count, values_offset, overlay, 0, 5000, ImGui::GetContentRegionAvail());

		ImGui::EndChild();
	}

	ImGui::SameLine();

	{
		ImGui::BeginChild("Child Right", ImVec2(0, 0));
		float image_height = ImGui::GetWindowHeight();
		ImGui::Image((void*)result_texture, { image_height, image_height }, { 0,0 }, { 1,1 }, { 1,1,1,1 }, { 1,1,1,1 });
		ImGui::EndChild();
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
	glfwSwapInterval(1);

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
