#version 330

#define float2 vec2
#define float3 vec3
#define float4 vec4
#define float4x4 mat4
#define float3x3 mat3

in float2 fragment_tex_coord;

layout(location = 0) out vec4 frag_color;

struct material {
  float3 amb;
  float3 dif;
  float3 spec;
  float shin;
};

uniform material emerald = material(
  float3(0.0215f, 0.1745f, 0.0215f),
  float3(0.07568f, 0.61424f, 0.07568f),
  float3(0.633f, 0.727811f, 0.633f),
  0.6f
);

uniform material ruby = material(
  float3(0.1745f, 0.01175f, 0.01175f),
  float3(0.61424f, 0.04136f, 0.04136f),
  float3(0.727811f, 0.626959f, 0.626959f),
  0.6f
);

uniform material yellow_rubber = material(
  float3(0.05f, 0.05f, 0.0f),
  float3(0.5f, 0.5f, 0.4f),
  float3(0.7f, 0.7f, 0.04f),
  0.078125f
);

uniform int g_screen_width;
uniform int g_screen_height;

uniform float4x4 g_ray_matrix;

uniform float4 g_bg_color = float4(0.0f, 0.0f, 0.3f, 1.0f);

uniform int max_steps = 100;
uniform float eps = 0.005f;

uniform float3 cam_pos;

uniform float attr_const_0 = 1.0f;
uniform float attr_const_1 = 0.0f;
uniform float attr_const_2 = 0.01f;

uniform float3 light_1_pos = float3(0.0f, 0.0f, 0.0f);
uniform float3 light_2_pos = float3(10.0f, 10.0f, 10.0f);

uniform float3 sphere_1_center = float3(-2.0f, -2.0f, 0.0f);
uniform float sphere_1_radius = 1.0f;

uniform float3 sphere_2_center = float3(-4.0f, -1.0f, 10.0f);
uniform float sphere_2_radius = 3.0f;

uniform float3 sphere_3_center = float3(-4.0f, 10.0f, 11.0f);
uniform float sphere_3_radius = 1.0f;

uniform float3 box_1_center = float3(2.0f, 2.0f, 2.0f);
uniform float3 box_1_size = float3(1.0f, 1.0f, 1.0f);

uniform float3 torus_1_center = float3(3.0f, 3.0f, 4.0f);
uniform float2 torus_1_t = float2(2.0f, 1.0f);

float3 eye_ray_dir(float x, float y, float w, float h)
{
	float fov = 3.141592654f/(2.0f); 
  float3 ray_dir;
  
	ray_dir.x = x + 0.5f - (w / 2.0f);
	ray_dir.y = y + 0.5f - (h / 2.0f);
	ray_dir.z = -(w) / tan(fov / 2.0f);
	
  return normalize(ray_dir);
}

float sd_sphere(float3 pos, float3 center,  float radius)
{
  return length(pos + center)-radius;
}

float ud_box(float3 pos, float3 center, float3 size) 
{
  return length(max(abs(pos + center) - size, 0.0f));
}

float sd_torus(float3 pos, float3 center, float2 t)
{
  float2 q = float2(length(pos.xy - center.xy)-t.x, pos.z - center.z);
  return length(q) - t.y;
}

float3 estimate_normal_sphere(float3 z, float3 center, float radius)
{
  float eps = 0.001f;
  float3 z1 = z + float3(eps, 0.0f, 0.0f);
  float3 z2 = z - float3(eps, 0.0f, 0.0f);
  float3 z3 = z + float3(0.0f, eps, 0.0f);
  float3 z4 = z - float3(0.0f, eps, 0.0f);
  float3 z5 = z + float3(0.0f, 0.0f, eps);
  float3 z6 = z - float3(0.0f, 0.0f, eps);
  float dx = sd_sphere(z1, center, radius) - sd_sphere(z2, center, radius);
  float dy = sd_sphere(z3, center, radius) - sd_sphere(z4, center, radius);
  float dz = sd_sphere(z5, center, radius) - sd_sphere(z6, center, radius);
  return normalize(float3(dx, dy, dz) / (2.0f * eps));
}

float3 estimate_normal_box_1(float3 z)
{
  float eps = 0.001f;
  float3 z1 = z + float3(eps, 0.0f, 0.0f);
  float3 z2 = z - float3(eps, 0.0f, 0.0f);
  float3 z3 = z + float3(0.0f, eps, 0.0f);
  float3 z4 = z - float3(0.0f, eps, 0.0f);
  float3 z5 = z + float3(0.0f, 0.0f, eps);
  float3 z6 = z - float3(0.0f, 0.0f, eps);
  float dx = ud_box(z1, box_1_center, box_1_size) - ud_box(z2, box_1_center, box_1_size);
  float dy = ud_box(z3, box_1_center, box_1_size) - ud_box(z4, box_1_center, box_1_size);
  float dz = ud_box(z5, box_1_center, box_1_size) - ud_box(z6, box_1_center, box_1_size);
  return normalize(float3(dx, dy, dz) / (2.0f * eps));
}

float3 estimate_normal_torus_1(float3 z)
{
  float eps = 0.001f;
  float3 z1 = z + float3(eps, 0.0f, 0.0f);
  float3 z2 = z - float3(eps, 0.0f, 0.0f);
  float3 z3 = z + float3(0.0f, eps, 0.0f);
  float3 z4 = z - float3(0.0f, eps, 0.0f);
  float3 z5 = z + float3(0.0f, 0.0f, eps);
  float3 z6 = z - float3(0.0f, 0.0f, eps);
  float dx = sd_torus(z1, torus_1_center, torus_1_t) - sd_torus(z2, torus_1_center, torus_1_t);
  float dy = sd_torus(z3, torus_1_center, torus_1_t) - sd_torus(z4, torus_1_center, torus_1_t);
  float dz = sd_torus(z5, torus_1_center, torus_1_t) - sd_torus(z6, torus_1_center, torus_1_t);
  return normalize(float3(dx, dy, dz) / (2.0f * eps));
}

float4 ray_trace(float3 ray_pos, float3 ray_dir) {
  float3 color = g_bg_color.xyz;
  float3 inter_point = ray_pos;
  float3 temp_pos = ray_pos;
  float step = 1000000;

  for (int i = 0; i < max_steps; i++) {
    step = sd_sphere(temp_pos, sphere_1_center, sphere_1_radius);
    if (step < eps) {
      float attr_1 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);
      float attr_2 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);

      float3 ambient = emerald.amb;

      float3 normal = -estimate_normal_sphere(inter_point, sphere_1_center, sphere_1_radius);
      float3 light_1_dir = -normalize(light_1_pos - inter_point);
      float3 light_2_dir = -normalize(light_2_pos - inter_point);
      float diff = max(dot(normal, light_1_dir), 0.0f) / attr_1
          + max(dot(normal, light_2_dir), 0.0f) / attr_2;
      float3 diffuse = diff * emerald.dif;

      float3 reflect_dir_1 = normalize(reflect(-light_1_dir, normal));
      float3 reflect_dir_2 = normalize(reflect(-light_2_dir, normal));
      float spec = pow(max(dot(ray_dir, reflect_dir_1), 0.0f), 128.0f * emerald.shin) / attr_1
          + pow(max(dot(ray_dir, reflect_dir_2), 0.0f), 128.0f * emerald.shin) / attr_2;
      float3 specular = spec * emerald.spec;

      color = ambient + diffuse + specular;
      return float4(color, 1.0f);
    }
    if (sd_sphere(temp_pos, sphere_2_center, sphere_2_radius) < step) {
      step = sd_sphere(temp_pos, sphere_2_center, sphere_2_radius);
      if (step < eps) {
        float attr_1 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);
        float attr_2 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);

        float3 ambient = ruby.amb;

        float3 normal = -estimate_normal_sphere(inter_point, sphere_2_center, sphere_2_radius);
        float3 light_1_dir = -normalize(light_1_pos - inter_point);
        float3 light_2_dir = -normalize(light_2_pos - inter_point);
        float diff = max(dot(normal, light_1_dir), 0.0f) / attr_1
            + max(dot(normal, light_2_dir), 0.0f) / attr_2;
        float3 diffuse = diff * ruby.dif;

        float3 reflect_dir_1 = normalize(reflect(-light_1_dir, normal));
        float3 reflect_dir_2 = normalize(reflect(-light_2_dir, normal));
        float spec = pow(max(dot(ray_dir, reflect_dir_1), 0.0f), 128.0f * ruby.shin) / attr_1
            + pow(max(dot(ray_dir, reflect_dir_2), 0.0f), 128.0f * ruby.shin) / attr_2;
        float3 specular = spec * ruby.spec;

        color = ambient + diffuse + specular;
        return float4(color, 1.0f);
      }
    }
    if (sd_sphere(temp_pos, sphere_3_center, sphere_3_radius) < step) {
      step = sd_sphere(temp_pos, sphere_3_center, sphere_3_radius);
      if (step < eps) {
        float attr_1 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);
        float attr_2 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);

        float3 ambient = ruby.amb;

        float3 normal = -estimate_normal_sphere(inter_point, sphere_3_center, sphere_3_radius);
        float3 light_1_dir = -normalize(light_1_pos - inter_point);
        float3 light_2_dir = -normalize(light_2_pos - inter_point);
        float diff = max(dot(normal, light_1_dir), 0.0f) / attr_1
            + max(dot(normal, light_2_dir), 0.0f) / attr_2;
        float3 diffuse = diff * yellow_rubber.dif;

        float3 reflect_dir_1 = normalize(reflect(-light_1_dir, normal));
        float3 reflect_dir_2 = normalize(reflect(-light_2_dir, normal));
        float spec = pow(max(dot(ray_dir, reflect_dir_1), 0.0f), 128.0f * yellow_rubber.shin) / attr_1
            + pow(max(dot(ray_dir, reflect_dir_2), 0.0f), 128.0f * yellow_rubber.shin) / attr_2;
        float3 specular = spec * yellow_rubber.spec;

        color = ambient + diffuse + specular;
        return float4(color, 1.0f);
      }
    }
    if (ud_box(temp_pos, box_1_center, box_1_size) < step) {
      step = ud_box(temp_pos, box_1_center, box_1_size);
      if (step < eps) {
        float attr_1 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);
        float attr_2 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);

        float3 ambient = ruby.amb;

        float3 normal = -estimate_normal_box_1(inter_point);
        float3 light_1_dir = -normalize(light_1_pos - inter_point);
        float3 light_2_dir = -normalize(light_2_pos - inter_point);
        float diff = max(dot(normal, light_1_dir), 0.0f) / attr_1
            + max(dot(normal, light_2_dir), 0.0f) / attr_2;
        float3 diffuse = diff * ruby.dif;

        float3 reflect_dir_1 = normalize(reflect(-light_1_dir, normal));
        float3 reflect_dir_2 = normalize(reflect(-light_2_dir, normal));
        float spec = pow(max(dot(ray_dir, reflect_dir_1), 0.0f), 128.0f * ruby.shin) / attr_1
            + pow(max(dot(ray_dir, reflect_dir_2), 0.0f), 128.0f * ruby.shin) / attr_2;
        float3 specular = spec * ruby.spec;
        
        color = ambient + diffuse + specular;
        return float4(color, 1.0f);
      }
    }
    if (sd_torus(temp_pos, torus_1_center, torus_1_t) < step) {
      step = sd_torus(temp_pos, torus_1_center, torus_1_t);
      if (step < eps) {
        float attr_1 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);
        float attr_2 = attr_const_0
          + attr_const_1 * length(light_1_pos - inter_point)
          + attr_const_2 * length(light_1_pos - inter_point) * length(light_1_pos - inter_point);

        float3 ambient = yellow_rubber.amb;

        float3 normal = -estimate_normal_torus_1(inter_point);
        float3 light_1_dir = -normalize(light_1_pos - inter_point);
        float3 light_2_dir = -normalize(light_2_pos - inter_point);
        float diff = max(dot(normal, light_1_dir), 0.0f) / attr_1 
            + max(dot(normal, light_2_dir), 0.0f) / attr_2;
        float3 diffuse = diff * yellow_rubber.dif;

        float3 reflect_dir_1 = normalize(reflect(-light_1_dir, normal));
        float3 reflect_dir_2 = normalize(reflect(-light_2_dir, normal));
        float spec = pow(max(dot(ray_dir, reflect_dir_1), 0.0f), 128.0f * yellow_rubber.shin) / attr_1
            + pow(max(dot(ray_dir, reflect_dir_2), 0.0f), 128.0f * yellow_rubber.shin) / attr_2;
        float3 specular = spec * yellow_rubber.spec;
        
        color = ambient + diffuse + specular;
        return float4(color, 1.0f);
      }
    }

    temp_pos = temp_pos + ray_dir * step / length(ray_dir);
    inter_point = temp_pos;
  }

  return float4(color, 1.0f);
}

void main(void)
{
  float w = float(g_screen_width);
  float h = float(g_screen_height);
  
  // get curr pixelcoordinates
  float x = fragment_tex_coord.x * w; 
  float y = fragment_tex_coord.y * h;
  
  // generate initial ray
  float3 ray_pos = cam_pos; 
  float3 ray_dir = eye_ray_dir(x,y,w,h);
 
  // transorm ray with matrix
  ray_dir = float3x3(g_ray_matrix) * ray_dir;
 
  frag_color = ray_trace(ray_pos, ray_dir);
}