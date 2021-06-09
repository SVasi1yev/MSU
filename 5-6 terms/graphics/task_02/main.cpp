//internal includes
#include "common.h"
#include "ShaderProgram.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//External dependencies
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <random>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>

//Window size
const GLsizei WIDTH = 1270, HEIGHT = 720;
const float pi = 3.14;
GLfloat delta_time = 0.0f;
GLfloat last_time = 0.0f;

int initGL()
{
    int res = 0;

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }

    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    return 0;
}

int main() 
{
    if (!glfwInit()) {
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL basic sample", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (initGL() != 0) {
        return -1;
    }

    GLenum gl_error = glGetError();
	while (gl_error != GL_NO_ERROR) {
		gl_error = glGetError();
    }
    
    glEnable(GL_DEPTH_TEST);

    std::unordered_map<GLenum, std::string> coub_shaders;
    coub_shaders[GL_VERTEX_SHADER] = "shaders/vertex.glsl";
    coub_shaders[GL_FRAGMENT_SHADER] = "shaders/fragment.glsl";
    ShaderProgram coub_program(coub_shaders); GL_CHECK_ERRORS;

    float coub_vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };

    float3 cube_pos[] = {
        float3(0.0f, 0.0f, 0.0f),
        float3(-1.0f, -1.0f, -4.0f),
        float3(-2.0f, 2.0f, -7.0f),
        float3(2.5f, -2.5f, -10.0f),
        float3(3.0f, 3.0f, -13.0f),
    };

    GLuint coub_VBO, coub_VAO;
    glGenVertexArrays(1, &coub_VAO);
    glGenBuffers(1, &coub_VBO);
    glBindVertexArray(coub_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, coub_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(coub_vertices), coub_vertices, GL_STATIC_DRAW);
    GL_CHECK_ERRORS;
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    GL_CHECK_ERRORS;

    unsigned int texture1;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int width, height, nrChannels;
    std::string path = "textures/container.jpg";
    GL_CHECK_ERRORS;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        auto type = (nrChannels == 3) ? GL_RGB : GL_RGBA;
        glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, type, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    GL_CHECK_ERRORS;


    std::unordered_map<GLenum, std::string> triangles_shaders;
    triangles_shaders[GL_VERTEX_SHADER] = "shaders/tr_vertex.glsl";
    triangles_shaders[GL_FRAGMENT_SHADER] = "shaders/tr_fragment.glsl";
    ShaderProgram triangles_program(triangles_shaders);

    float triangles_vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    };

    float4 colors[] = {
        float4(1.0f, 0.0f, 0.0f, 0.5f),
        float4(0.0f, 1.0f, 0.0f, 1.0f),
        float4(0.0f, 0.0f, 1.0f, 1.0f),
        float4(0.0f, 1.0f, 1.0f, 1.0f),
        float4(1.0f, 0.0f, 1.0f, 1.0f),
        float4(1.0f, 1.0f, 1.0f, 1.0f),
    };

    float3 triangles_pos[] = {
        float3(0.0f, 0.0f, 0.0f)
    };

    GLuint triangles_VBO, triangles_VAO;
    glGenVertexArrays(1, &triangles_VAO);
    glGenBuffers(1, &triangles_VBO);
    glBindVertexArray(triangles_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, triangles_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangles_vertices), triangles_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    float umbrella1_vertices[] = {
        -0.25, 0.5, 0.0,
        0.25, 0.5, 0.0,
        0.0, 0.0, 0.0,

        0.5, 0.25, 0.0,
        0.5, -0.25, 0.0,
        0.0, 0.0, 0.0,

        0.25, -0.5, 0.0,
        -0.25, -0.5, 0.0,
        0.0, 0.0, 0.0,

        -0.5, -0.25, 0.0,
        -0.5, 0.25, 0.0,
        0.0, 0.0, 0.0
    };

    GLuint umbrella1_VBO, umbrella1_VAO;
    glGenVertexArrays(1, &umbrella1_VAO);
    glGenBuffers(1, &umbrella1_VBO);
    glBindVertexArray(umbrella1_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, umbrella1_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(umbrella1_vertices), umbrella1_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    float umbrella2_vertices[] = {
        0.5, 0.25, 0.0,
        0.25, 0.5, 0.0,
        0.0, 0.0, 0.0,

        0.25, -0.5, 0.0,
        0.5, -0.25, 0.0,
        0.0, 0.0, 0.0,

        -0.5, -0.25, 0.0,
        -0.25, -0.5, 0.0,
        0.0, 0.0, 0.0,

        -0.25, 0.5, 0.0,
        -0.5, 0.25, 0.0,
        0.0, 0.0, 0.0
    };

    GLuint umbrella2_VBO, umbrella2_VAO;
    glGenVertexArrays(1, &umbrella2_VAO);
    glGenBuffers(1, &umbrella2_VBO);
    glBindVertexArray(umbrella2_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, umbrella2_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(umbrella2_vertices), umbrella2_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window))
    {   
        GLfloat cur_time = glfwGetTime();
        delta_time = cur_time - last_time;
        last_time = cur_time;
        glfwPollEvents(); 

        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);

        coub_program.StartUseShader();
        
        float4x4 view;
        float4x4 projection;        
        projection = projectionMatrixTransposed(45, (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f);
        view       = transpose(translate4x4(float3(0.0f, 0.0f, -10.0f)));
        coub_program.SetUniform("view", view);
        coub_program.SetUniform("projection", projection);
        
        glBindVertexArray(coub_VAO);
        for (unsigned int i = 1; i < 5; i++)
        {
            float4x4 model = transpose(translate4x4(cube_pos[i]));
            float rot_rate = 1.0; 
            float4x4 move = transpose(translate4x4(make_float3(1.5 * cos(cur_time), 0.0, 1.5 * sin(cur_time))));
            model = mul(rotate_X_4x4(cur_time * rot_rate), model);
            model = mul(rotate_Y_4x4(cur_time * rot_rate), model);
            model = mul(model, move);
            coub_program.SetUniform("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        coub_program.StopUseShader();

        triangles_program.StartUseShader();
        triangles_program.SetUniform("view", view);
        triangles_program.SetUniform("projection", projection);
        glBindVertexArray(triangles_VAO);
        for (int i = 0; i < 6; i++) {
            float4x4 model = transpose(translate4x4(triangles_pos[0]));
            float rot_rate = 1.0;
            float phasa = 2 * pi * i / 6;
            float4x4 move = transpose(translate4x4(make_float3(3.5 * cos(phasa + cur_time), 3.5 * sin(phasa + cur_time), 5 * sin(phasa + cur_time) - 5)));
            model = mul(rotate_Z_4x4(cur_time * rot_rate), model);
            model = mul(model, move);
            triangles_program.SetUniform("model", model);
            triangles_program.SetUniform("color", colors[i]);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        triangles_program.StopUseShader();

        triangles_program.StartUseShader();
        triangles_program.SetUniform("view", view);
        triangles_program.SetUniform("projection", projection);
        glBindVertexArray(umbrella1_VAO);
        float4x4 model = transpose(translate4x4(triangles_pos[0]));
        float rot_rate = 1.0;
        float4x4 move = transpose(translate4x4(make_float3(0.0, 0.0, 5 * sin(cur_time) - 5)));
        model = mul(rotate_Z_4x4(cur_time * rot_rate), model);
        model = mul(model, move);
        triangles_program.SetUniform("model", model);
        triangles_program.SetUniform("color", colors[0]);
        glDrawArrays(GL_TRIANGLES, 0, 12);
        triangles_program.StopUseShader();

        triangles_program.StartUseShader();
        triangles_program.SetUniform("view", view);
        triangles_program.SetUniform("projection", projection);
        glBindVertexArray(umbrella2_VAO);
        model = transpose(translate4x4(triangles_pos[0]));
        rot_rate = 1.0;
        move = transpose(translate4x4(make_float3(0.0, 0.0, 5 * sin(cur_time) - 5)));
        model = mul(rotate_Z_4x4(cur_time * rot_rate), model);
        model = mul(model, move);
        triangles_program.SetUniform("model", model);
        triangles_program.SetUniform("color", colors[5]);
        glDrawArrays(GL_TRIANGLES, 0, 12);
        triangles_program.StopUseShader();

        glfwSwapBuffers(window);
        
    }

    glDeleteVertexArrays(1, &coub_VAO);
    glDeleteBuffers(1, &coub_VBO);
    glfwTerminate();

    return 0;
}