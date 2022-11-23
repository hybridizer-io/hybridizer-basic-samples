namespace MigrateSamples
{
    internal class Program
    {
        const string INPUT_DIR = "HybridizerBasicSamples_CUDA116";
        const string TARGET_VER = "11.6";
        static readonly string[] OLD_VER = new string[]
        {
            "10.0", "10.1", "11.0"
        };
        const string OLD_PATH_AND_TOOLSET = """
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '16.0'">
    <MyToolset>v142</MyToolset>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.16.0)</HybridizerInstallPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '15.0'">
    <MyToolset>v141</MyToolset>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.15.0)</HybridizerInstallPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '14.0'">
    <MyToolset>v140</MyToolset>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.14.0)</HybridizerInstallPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '12.0'">
    <MyToolset>v120</MyToolset>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.12.0)</HybridizerInstallPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '11.0'">
    <MyToolset>v110</MyToolset>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.11.0)</HybridizerInstallPath>
  </PropertyGroup>
""";

        const string OLD_PATH_AND_TOOLSET_2 = """
    <PropertyGroup Condition="'$(VisualStudioVersion)' == '11.0'">
      <MyToolset>v110</MyToolset>
      <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.11.0)</HybridizerInstallPath>
    </PropertyGroup>
    <PropertyGroup Condition="'$(VisualStudioVersion)' == '12.0'">
      <MyToolset>v120</MyToolset>
      <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.12.0)</HybridizerInstallPath>
    </PropertyGroup>
    <PropertyGroup Condition="'$(VisualStudioVersion)' == '14.0'">
      <MyToolset>v140</MyToolset>
      <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.14.0)</HybridizerInstallPath>
    </PropertyGroup>
    <PropertyGroup Condition="'$(VisualStudioVersion)' == '15.0'">
      <MyToolset>v141</MyToolset>
      <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.15.0)</HybridizerInstallPath>
    </PropertyGroup>
    <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
    <PropertyGroup Condition="'$(VisualStudioVersion)' == '16.0'">
      <MyToolset>v142</MyToolset>
      <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath.16.0)</HybridizerInstallPath>
    </PropertyGroup>
  """;

        const string NEW_PATH_AND_TOOLSET_2= """
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath)</HybridizerInstallPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '17.0'">
    <MyToolset>v143</MyToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '16.0'">
    <MyToolset>v142</MyToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '15.0'">
    <MyToolset>v141</MyToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '14.0'">
    <MyToolset>v140</MyToolset>
  </PropertyGroup>
""";

        const string NEW_PATH_AND_TOOLSET = """
  <PropertyGroup>
    <HybridizerInstallPath>$(Registry:HKEY_CURRENT_USER\Software\ALTIMESH\Hybridizer@vsixInstallPath)</HybridizerInstallPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '17.0'">
    <MyToolset>v143</MyToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '16.0'">
    <MyToolset>v142</MyToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '15.0'">
    <MyToolset>v141</MyToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(VisualStudioVersion)' == '14.0'">
    <MyToolset>v140</MyToolset>
  </PropertyGroup>
""";

        const string COPY_RUNTIME_1 = @"echo copy ""$(HybridizerInstallPath)\bin\hybridizer.basic.runtime.dll"" ""$(OutDir)""";
        const string COPY_RUNTIME_2 = @"copy ""$(HybridizerInstallPath)\bin\hybridizer.basic.runtime.dll"" ""$(OutDir)""";


        static void Main(string[] args)
        {
            foreach (var vcxproj in Directory.GetFiles(INPUT_DIR, "*.vcxproj", SearchOption.AllDirectories))
            {
                string content = File.ReadAllText(vcxproj);
                content = content.Replace(COPY_RUNTIME_1, "");
                content = content.Replace(COPY_RUNTIME_2, "");
                content = content.Replace(OLD_PATH_AND_TOOLSET, NEW_PATH_AND_TOOLSET);
                content = content.Replace(OLD_PATH_AND_TOOLSET_2, NEW_PATH_AND_TOOLSET_2);
                content = content.Replace("CrossVersionToolset", "MyToolset");
                content = content.Replace("hybridizer.basic.runtime.lib;", "");
                foreach (var old_ver in OLD_VER)
                {
                    content = content.Replace($"PTXJitterService.{old_ver}.exe", $"PTXJitterService.{TARGET_VER}.exe");
                    content = content.Replace($"CUDA {old_ver}.targets", $"CUDA {TARGET_VER}.targets");
                    content = content.Replace($"CUDA {old_ver}.props", $"CUDA {TARGET_VER}.props");
                }

                File.WriteAllText(vcxproj, content);
            }

            foreach (var csproj in Directory.GetFiles(INPUT_DIR, "*.csproj", SearchOption.AllDirectories))
            {
                string content = File.ReadAllText(csproj);
                content = content.Replace(@"<Reference Include=""Hybridizer.Runtime.CUDAImports"" />",
    @"<Reference Include = ""Hybridizer.Runtime.CUDAImports, Version=1.0.0.0, Culture=neutral, PublicKeyToken=2e60b65856451e38, processorArchitecture=AMD64"" />");
                content = content.Replace("<TargetFrameworkVersion>v4.5.2</TargetFrameworkVersion>", @"<TargetFrameworkVersion>v4.8</TargetFrameworkVersion>");

                File.WriteAllText(csproj, content);
            }
        }
    }
}