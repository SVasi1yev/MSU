#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 frag_color;

uniform vec3 color;
uniform vec3 light_pos;
uniform vec3 view_pos;

void main()
{
    float ambStr = 0.1f;

    vec3 norm = normalize(Normal);
    vec3 light_dir = normalize(light_pos - FragPos);

    float diff = max(dot(norm, light_dir), 0.0);

    float specStr = 0.7f;
    vec3 view_dir = normalize(view_pos - FragPos);
    vec3 reflect_dir = -reflect(-light_dir, norm);
    float spec = specStr * pow(max(dot(view_dir, reflect_dir), 0.0), 128);
    
    vec3 result = (ambStr + diff + spec) * color;

    frag_color = vec4(result, 1.0f);
}