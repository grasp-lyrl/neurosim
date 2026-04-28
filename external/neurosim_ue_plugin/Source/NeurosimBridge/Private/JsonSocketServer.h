// Copyright Neurosim contributors. Licensed under the repo LICENSE.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"

class FJsonObject;

/**
 * AF_UNIX SOCK_STREAM server that speaks length-prefixed JSON
 * ([uint32 BE length][utf-8 body]).
 *
 * One client at a time (Python driver). The accept loop runs on a background
 * thread; each method handler dispatches the actual work to the game thread
 * via AsyncTask(ENamedThreads::GameThread, ...) and blocks on a completion
 * event so responses stay in-order.
 */
class FJsonSocketServer : public FRunnable
{
public:
    FJsonSocketServer();
    virtual ~FJsonSocketServer();

    /** Bind + listen. Returns false on error (path in use, insufficient perms, ...). */
    bool Start(const FString& SocketPath);

    /** Stop the accept loop, close all fds, join the worker thread. */
    void Stop();

    // FRunnable
    virtual uint32 Run() override;
    virtual void Stop_Internal() { bStopRequested.store(true); }
    virtual void Exit() override {}

private:
    bool ReadFrame(int ClientFd, TArray<uint8>& OutBytes);
    bool WriteFrame(int ClientFd, const TArray<uint8>& Bytes);

    /** Main per-client loop. One request in, one response out. */
    void ServeClient(int ClientFd);

    /** Dispatch a parsed request to the correct handler. */
    TSharedPtr<FJsonObject> Dispatch(const TSharedPtr<FJsonObject>& Request);

    // Handlers. Each reads request params, does work on the game thread, and
    // returns the "result" sub-object (or nullptr on error, with *OutError set).
    TSharedPtr<FJsonObject> HandleHandshake(const TSharedPtr<FJsonObject>& Params, FString& OutError);
    TSharedPtr<FJsonObject> HandleSetAgentPose(const TSharedPtr<FJsonObject>& Params, FString& OutError);
    TSharedPtr<FJsonObject> HandleGetAgentPose(const TSharedPtr<FJsonObject>& Params, FString& OutError);
    TSharedPtr<FJsonObject> HandleOpenLevel(const TSharedPtr<FJsonObject>& Params, FString& OutError);
    TSharedPtr<FJsonObject> HandleRenderFrame(const TSharedPtr<FJsonObject>& Params, FString& OutError);
    TSharedPtr<FJsonObject> HandleShutdown(const TSharedPtr<FJsonObject>& Params, FString& OutError);

    FString BoundPath;
    int ListenFd = -1;
    TUniquePtr<FRunnableThread> Thread;
    std::atomic<bool> bStopRequested{false};
};
