// Copyright Neurosim contributors. Licensed under the repo LICENSE.

using UnrealBuildTool;

public class NeurosimBridge : ModuleRules
{
    public NeurosimBridge(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "RenderCore",
            "RHI",
            "Sockets",
            "Networking",
            "Json",
            "JsonUtilities",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate",
            "SlateCore",
        });

        // Phase 2+: Vulkan RHI access for VK_KHR_external_memory_fd.
        // PrivateDependencyModuleNames.Add("VulkanRHI");
        // AddEngineThirdPartyPrivateStaticDependencies(Target, "Vulkan");

        if (Target.Platform != UnrealTargetPlatform.Linux)
        {
            throw new BuildException("NeurosimBridge is Linux-only.");
        }
    }
}
