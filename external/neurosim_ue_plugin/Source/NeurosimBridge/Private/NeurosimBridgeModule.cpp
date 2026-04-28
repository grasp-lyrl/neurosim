// Copyright Neurosim contributors. Licensed under the repo LICENSE.

#include "NeurosimBridge.h"
#include "JsonSocketServer.h"

#include "Misc/CommandLine.h"
#include "Misc/Parse.h"

DEFINE_LOG_CATEGORY_STATIC(LogNeurosimBridge, Log, All);

IMPLEMENT_MODULE(FNeurosimBridgeModule, NeurosimBridge);

FNeurosimBridgeModule& FNeurosimBridgeModule::Get()
{
    return FModuleManager::LoadModuleChecked<FNeurosimBridgeModule>("NeurosimBridge");
}

void FNeurosimBridgeModule::StartupModule()
{
    UE_LOG(LogNeurosimBridge, Log, TEXT("NeurosimBridge: StartupModule"));

    // Socket path is passed from Python via -NeurosimSocket=<path>. If absent, we do
    // not start the server -- allows the packaged build to be launched manually for
    // smoke-testing without a Python driver attached.
    FString SocketPath;
    if (!FParse::Value(FCommandLine::Get(), TEXT("NeurosimSocket="), SocketPath))
    {
        UE_LOG(LogNeurosimBridge, Warning,
               TEXT("NeurosimBridge: -NeurosimSocket=<path> not provided; server disabled."));
        return;
    }

    SocketServer = MakeUnique<FJsonSocketServer>();
    if (!SocketServer->Start(SocketPath))
    {
        UE_LOG(LogNeurosimBridge, Error,
               TEXT("NeurosimBridge: failed to start server on %s"), *SocketPath);
        SocketServer.Reset();
        return;
    }

    UE_LOG(LogNeurosimBridge, Log,
           TEXT("NeurosimBridge: server listening on %s"), *SocketPath);
}

void FNeurosimBridgeModule::ShutdownModule()
{
    UE_LOG(LogNeurosimBridge, Log, TEXT("NeurosimBridge: ShutdownModule"));
    if (SocketServer.IsValid())
    {
        SocketServer->Stop();
        SocketServer.Reset();
    }
}
