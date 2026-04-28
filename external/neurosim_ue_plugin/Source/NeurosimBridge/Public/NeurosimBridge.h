// Copyright Neurosim contributors. Licensed under the repo LICENSE.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FJsonSocketServer;

class FNeurosimBridgeModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    static FNeurosimBridgeModule& Get();

    /** Active socket server (valid between StartupModule and ShutdownModule). */
    FJsonSocketServer* GetSocketServer() const { return SocketServer.Get(); }

private:
    TUniquePtr<FJsonSocketServer> SocketServer;
};
