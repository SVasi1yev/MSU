//internal includes
#include "common.h"
#include "ShaderProgram.h"
#include "LiteMath.h"

//External dependencies
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <random>

static GLsizei WIDTH = 800, HEIGHT = 800; //размеры окна

using namespace LiteMath;

float3 camera_pos = float3(0.0f, 0.0f, 10.0f);
float3 camera_front = float3(0.0f, 0.0f, -1.0f);
float3 camera_up = float3(0.0f, 1.0f, 0.0f);
GLfloat last_x = 400, last_y = 300;

float3 g_camPos(0, 0, 5);
float  cam_rot[2] = {0,0};
int    mx = 0, my = 0;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
  GLfloat camera_speed = 0.2f;
  if(key == GLFW_KEY_W) {
    camera_pos += camera_speed * camera_front;
  }
  if(key == GLFW_KEY_S) {
    camera_pos -= camera_speed * camera_front;
  }
  if(key == GLFW_KEY_A) {
    camera_pos -= camera_speed * normalize(cross(camera_front, camera_up));
  }
  if(key == GLFW_KEY_D) {
    camera_pos += camera_speed * normalize(cross(camera_front, camera_up));
  }
}

static void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
  xpos *= -4.5f;
  ypos *= 4.5f;

  int x1 = int(xpos);
  int y1 = int(ypos);

  cam_rot[0] -= 0.0025f * (y1 - my);	//Изменение угола поворота
  cam_rot[1] -= 0.0025f * (x1 - mx);

  mx = int(xpos);
  my = int(ypos);
  camera_front = mul(mul(rotate_Y_4x4(-cam_rot[1]), rotate_X_4x4(+cam_rot[0])), float3(0.0f, 0.0f, -1.0f));
  camera_up = mul(mul(rotate_Y_4x4(-cam_rot[1]), rotate_X_4x4(+cam_rot[0])), float3(0.0f, 1.0f, 0.0f));
}

void window_resize(GLFWwindow* window, int width, int height)
{
  WIDTH  = width;
  HEIGHT = height;
}

int init_GL()
{
	int res = 0;
	//грузим функции opengl через glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	std::cout << "Vendor: "   << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: "  << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL: "     << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	return 0;
}

int main(int argc, char** argv)
{
	if(!glfwInit())
    return -1;

	//запрашиваем контекст opengl версии 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); 
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE); 

  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL ray marching sample", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window); 
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetKeyCallback(window, key_callback);
  glfwSetWindowSizeCallback(window, window_resize);

	if(init_GL() != 0) 
		return -1;
	
  //Reset any OpenGL errors which could be present for some reason
	GLenum gl_error = glGetError();
	while (gl_error != GL_NO_ERROR)
		gl_error = glGetError();

	//создание шейдерной программы из двух файлов с исходниками шейдеров
	//используется класс-обертка ShaderProgram
	std::unordered_map<GLenum, std::string> shaders;
	shaders[GL_VERTEX_SHADER]   = "vertex.glsl";
	shaders[GL_FRAGMENT_SHADER] = "fragment.glsl";
	ShaderProgram program(shaders); GL_CHECK_ERRORS;

  glfwSwapInterval(1); // force 60 frames per second
  
  //Создаем и загружаем геометрию поверхности
  //
  GLuint g_vertex_buffer_object;
  GLuint g_vertex_array_object;
  {
    float quad_pos[] =
    {
      -1.0f,  1.0f,	// v0 - top left corner
      -1.0f, -1.0f,	// v1 - bottom left corner
      1.0f,  1.0f,	// v2 - top right corner
      1.0f, -1.0f	  // v3 - bottom right corner
    };

    g_vertex_buffer_object = 0;
    GLuint vertexLocation = 0; // simple layout, assume have only positions at location = 0

    glGenBuffers(1, &g_vertex_buffer_object);                                                        GL_CHECK_ERRORS;
    glBindBuffer(GL_ARRAY_BUFFER, g_vertex_buffer_object);                                           GL_CHECK_ERRORS;
    glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), (GLfloat*)quad_pos, GL_STATIC_DRAW);      GL_CHECK_ERRORS;

    glGenVertexArrays(1, &g_vertex_array_object);                                                    GL_CHECK_ERRORS;
    glBindVertexArray(g_vertex_array_object);                                                        GL_CHECK_ERRORS;

    glBindBuffer(GL_ARRAY_BUFFER, g_vertex_buffer_object);                                           GL_CHECK_ERRORS;
    glEnableVertexAttribArray(vertexLocation);                                                       GL_CHECK_ERRORS;
    glVertexAttribPointer(vertexLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);                              GL_CHECK_ERRORS;

    glBindVertexArray(0);
  }

	//цикл обработки сообщений и отрисовки сцены каждый кадр
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		//очищаем экран каждый кадр
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);               GL_CHECK_ERRORS;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); GL_CHECK_ERRORS;

    program.StartUseShader();                           GL_CHECK_ERRORS;

    float4x4 cam_rot_matrix   = mul(rotate_Y_4x4(-cam_rot[1]), rotate_X_4x4(+cam_rot[0]));
    float4x4 cam_trans_matrix = translate4x4(g_camPos);
    float4x4 ray_matrix      = mul(cam_rot_matrix, cam_trans_matrix);
    program.SetUniform("g_ray_matrix", cam_rot_matrix);
    program.SetUniform("cam_pos", camera_pos);
    
    //program.SetUniform("cam_front", camera_front);

    program.SetUniform("g_screen_width" , WIDTH);
    program.SetUniform("g_screen_height", HEIGHT);

    // очистка и заполнение экрана цветом
    //
    glViewport  (0, 0, WIDTH, HEIGHT);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // draw call
    //
    glBindVertexArray(g_vertex_array_object); GL_CHECK_ERRORS;
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  GL_CHECK_ERRORS;  // The last parameter of glDrawArrays is equal to VS invocations
    
    program.StopUseShader();

		glfwSwapBuffers(window); 
	}

	//очищаем vboи vao перед закрытием программы
  //
	glDeleteVertexArrays(1, &g_vertex_array_object);
  glDeleteBuffers(1,      &g_vertex_buffer_object);

	glfwTerminate();
	return 0;
}