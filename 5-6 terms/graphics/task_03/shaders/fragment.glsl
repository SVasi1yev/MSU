#version 330 core
out vec4 frag_color;

in vec3 FragPos;
in vec2 TexCoord;
in vec3 Normal;

uniform sampler2D texture1;
uniform vec3 light_pos;
uniform vec3 view_pos;

void main()
{
    vec3 color = vec3(texture(texture1, TexCoord));

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