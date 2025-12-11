#version 400
out vec4 color;
in vec2 TexCoords;

uniform sampler2D text;

void main()
{
	color = texture(text, TexCoords);
}